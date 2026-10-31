import { TextGenerationError, type ProviderInstanceId } from "@t3tools/contracts";
import * as AcpSessionRuntime from "@t3tools/provider-acp/server/AcpSessionRuntime";
import * as Crypto from "effect/Crypto";
import * as Effect from "effect/Effect";
import * as Layer from "effect/Layer";
import * as Ref from "effect/Ref";
import * as ChildProcessSpawner from "effect/process/ChildProcessSpawner";

import type { AcpRegistrySettings } from "../settings.ts";
import * as AcpRegistrySupport from "./AcpRegistrySupport.ts";

// ACP models without a model switch report this placeholder slug; never send it to the agent.
const DEFAULT_MODEL_SLUG = "default";

const failure = (detail: string, cause?: unknown) =>
  new TextGenerationError({ operation: "describeImage", detail, cause });

export interface AcpRegistryImagePromptInput {
  readonly instanceId: ProviderInstanceId;
  readonly settings: AcpRegistrySettings;
  readonly cwd: string;
  readonly environment: NodeJS.ProcessEnv;
  readonly model: string;
  readonly prompt: string;
  readonly image: { readonly data: string; readonly mimeType: string };
}

/**
 * Runs one throwaway agent session: the agent's own login and model switch do the work,
 * and every permission request is refused so describing an image never runs a tool.
 */
export const promptAcpRegistryWithImage = Effect.fn("AcpRegistry.promptWithImage")(function* (
  input: AcpRegistryImagePromptInput,
) {
  const catalog = yield* AcpRegistrySupport.AcpRegistryCatalog;
  const childProcessSpawner = yield* ChildProcessSpawner.ChildProcessSpawner;
  const crypto = yield* Crypto.Crypto;
  return yield* Effect.gen(function* () {
    const resolved = yield* catalog
      .resolve(input.settings, input.cwd, input.environment)
      .pipe(Effect.mapError((cause) => failure("Could not start the ACP agent.", cause)));
    const runtimeContext = yield* Layer.build(
      AcpSessionRuntime.layer({
        spawn: resolved.spawn,
        cwd: input.cwd,
        clientCapabilities: {
          fs: { readTextFile: false, writeTextFile: false },
          terminal: false,
        },
        clientInfo: { name: "t3-code-image-relay", version: "0.0.0" },
        authenticateOnAuthRequired: false,
        ...(input.settings.authMethodId ? { authMethodId: input.settings.authMethodId } : {}),
      }).pipe(
        Layer.provide(
          Layer.mergeAll(
            Layer.succeed(ChildProcessSpawner.ChildProcessSpawner, childProcessSpawner),
            Layer.succeed(Crypto.Crypto, crypto),
          ),
        ),
      ),
    );
    const runtime = yield* Effect.service(AcpSessionRuntime.AcpSessionRuntime).pipe(
      Effect.provide(runtimeContext),
    );
    const reply = yield* Ref.make("");
    yield* runtime.handleRequestPermission(() =>
      Effect.succeed({ outcome: { outcome: "cancelled" } } as const),
    );
    yield* runtime.handleSessionUpdate((notification) => {
      const update = notification.update;
      if (update.sessionUpdate !== "agent_message_chunk") return Effect.void;
      const content = update.content;
      return content.type === "text"
        ? Ref.update(reply, (text) => text + content.text)
        : Effect.void;
    });
    const started = yield* runtime
      .start()
      .pipe(Effect.mapError((cause) => failure("The ACP agent could not open a session.", cause)));
    if (started.initializeResult.agentCapabilities?.promptCapabilities?.image !== true) {
      return yield* failure("This ACP agent does not accept images.");
    }
    const modelConfigId = started.modelConfigId ?? "model";
    const previousModel = (yield* runtime.getConfigOptions).find(
      (option) => option.id === modelConfigId && typeof option.currentValue === "string",
    )?.currentValue;
    const switchModel = input.model !== DEFAULT_MODEL_SLUG && input.model !== previousModel;
    if (switchModel) {
      yield* runtime
        .setModel(input.model)
        .pipe(
          Effect.mapError((cause) =>
            failure(`The ACP agent could not switch to model '${input.model}'.`, cause),
          ),
        );
    }
    const ask = runtime
      .prompt({
        prompt: [
          { type: "text", text: input.prompt },
          { type: "image", data: input.image.data, mimeType: input.image.mimeType },
        ],
      })
      .pipe(Effect.mapError((cause) => failure("The ACP agent failed to answer.", cause)));
    // Agents such as Mistral Vibe save a model switch as the user's default; put it back.
    yield* switchModel && typeof previousModel === "string"
      ? ask.pipe(Effect.ensuring(runtime.setModel(previousModel).pipe(Effect.ignore)))
      : ask;
    return yield* Ref.get(reply);
  }).pipe(
    Effect.scoped,
    Effect.mapError((error) =>
      error._tag === "TextGenerationError" ? error : failure("The ACP agent failed.", error),
    ),
  );
});
