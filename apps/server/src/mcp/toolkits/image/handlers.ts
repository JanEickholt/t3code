import { OrchestratorMcpFailure } from "@t3tools/contracts";
import * as Effect from "effect/Effect";
import * as Option from "effect/Option";
import * as ImageRelay from "../../../provider/ImageRelay.ts";
import * as ThreadManagementService from "../../../orchestration-v2/ThreadManagementService.ts";
import { threadWorkspaceRoot } from "../../../project/threadWorkspaceRoot.ts";
import * as McpInvocationContext from "../../McpInvocationContext.ts";
import * as McpToolAccess from "../../McpToolAccess.ts";
import { unavailable } from "../../threadAccess.ts";
import { ImageToolkit } from "./tools.ts";

export const layer = McpToolAccess.toLayer(
  ImageToolkit,
  Effect.succeed({
    t3_image_describe: McpToolAccess.readsAsCaller((input) =>
      Effect.gen(function* () {
        const scope = yield* McpInvocationContext.McpInvocationContext;
        const threadId = scope.thread?.threadId;
        if (threadId === undefined) return yield* unavailable();
        if (!scope.capabilities.has("image-relay")) {
          return yield* new OrchestratorMcpFailure({
            code: "capability_denied",
            message: "This thread's model can read images itself.",
          });
        }
        const threads = yield* ThreadManagementService.ThreadManagementService;
        const shell = yield* threads.getThreadShell(threadId).pipe(Effect.mapError(unavailable));
        if (shell === null) return yield* unavailable();
        const cwd = yield* threadWorkspaceRoot(threadId);
        // Optional so MCP hosts without provider text generation still list the tool.
        const relay = yield* Effect.serviceOption(ImageRelay.ImageRelay);
        if (Option.isNone(relay)) {
          return yield* new OrchestratorMcpFailure({
            code: "invalid_request",
            message: "Image description is unavailable on this server.",
          });
        }
        const transcription = yield* relay.value
          .describeFile({
            path: input.path,
            question: input.question,
            projectId: shell.projectId,
            cwd: cwd ?? null,
          })
          .pipe(
            Effect.mapError(
              (error) =>
                new OrchestratorMcpFailure({ code: "invalid_request", message: error.detail }),
            ),
          );
        return { transcription };
      }),
    ),
  }),
);
