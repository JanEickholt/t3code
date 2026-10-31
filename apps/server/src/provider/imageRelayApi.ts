import { type ImageRelayApi, TextGenerationError } from "@t3tools/contracts";
import { imageDescriptionInstructions } from "@t3tools/provider-core/server/textGenerationPrompts";
import * as Effect from "effect/Effect";
import * as Schema from "effect/Schema";
import { HttpClient, HttpClientRequest, HttpClientResponse } from "effect/http";

const ChatCompletion = Schema.Struct({
  choices: Schema.Array(
    Schema.Struct({ message: Schema.Struct({ content: Schema.NullOr(Schema.String) }) }),
  ),
});

const failure = (detail: string, cause?: unknown) =>
  new TextGenerationError({ operation: "describeImage", detail, cause });

/** `host/model`, so logs and saved transcriptions name the endpoint without its key. */
export function imageRelayApiLabel(api: Pick<ImageRelayApi, "baseUrl" | "model">): string {
  const host = URL.canParse(api.baseUrl) ? new URL(api.baseUrl).host : api.baseUrl;
  return `${host}/${api.model}`;
}

/** Plain text, not JSON: a one-key JSON wrapper lets models file the transcript under other keys. */
export const describeImageViaApi = Effect.fn("ImageRelayApi.describe")(function* (input: {
  readonly api: ImageRelayApi;
  readonly name: string;
  readonly mimeType: string;
  readonly bytes: Uint8Array;
  readonly question?: string | undefined;
}) {
  const client = yield* HttpClient.HttpClient;
  const url = `${input.api.baseUrl.replace(/\/+$/, "")}/chat/completions`;
  const text = [
    imageDescriptionInstructions({ name: input.name, question: input.question }),
    "",
    "Reply with the transcription as plain text.",
  ].join("\n");
  const request = yield* HttpClientRequest.post(url).pipe(
    input.api.apiKey.length > 0
      ? HttpClientRequest.bearerToken(input.api.apiKey)
      : (request) => request,
    HttpClientRequest.bodyJson({
      model: input.api.model,
      temperature: 0,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text },
            {
              type: "image_url",
              image_url: {
                url: `data:${input.mimeType};base64,${Buffer.from(input.bytes).toString("base64")}`,
              },
            },
          ],
        },
      ],
    }),
    Effect.mapError((cause) => failure("Could not encode the image request.", cause)),
  );
  const response = yield* client
    .execute(request)
    .pipe(Effect.mapError((cause) => failure(`Could not reach ${url}.`, cause)));
  if (response.status < 200 || response.status >= 300) {
    const body = yield* response.text.pipe(Effect.orElseSucceed(() => ""));
    return yield* failure(
      `The image API answered ${response.status}${body.length > 0 ? `: ${body.slice(0, 200)}` : "."}`,
    );
  }
  const completion = yield* HttpClientResponse.schemaBodyJson(ChatCompletion)(response).pipe(
    Effect.mapError((cause) => failure("The image API returned an unexpected response.", cause)),
  );
  const description = completion.choices[0]?.message.content?.trim() ?? "";
  if (description.length === 0) {
    return yield* failure("The image API returned an empty description.");
  }
  return description;
});
