import { assert, describe, it } from "@effect/vitest";
import { ChatAttachmentId, ProviderInstanceId } from "@t3tools/contracts";
import * as Effect from "effect/Effect";

import { fromRunner, type Request, type Runner } from "./textGenerationOperations.ts";

const image = {
  type: "image" as const,
  id: ChatAttachmentId.make("thread-1-0b5b8a1e-6f0a-4e0e-9a43-1f2b3c4d5e6f-png"),
  name: "error.png",
  mimeType: "image/png",
  sizeBytes: 4,
};
const modelSelection = { instanceId: ProviderInstanceId.make("codex"), model: "gpt-6-luna" };

describe("fromRunner describeImage", () => {
  it("is absent unless the runner delivers images", () => {
    const run: Runner = () => Effect.die("unused");
    assert.isUndefined(fromRunner("Test", run).describeImage);
  });

  it.effect("sends the image to the runner and rejects an empty description", () =>
    Effect.gen(function* () {
      const requests: Array<Request<never>> = [];
      let description = "  A stack trace.  ";
      const run = ((request: Request<never>) => {
        requests.push(request);
        return Effect.succeed({ description });
      }) as unknown as Runner;
      const describeImage = fromRunner("Test", run, { images: true }).describeImage!;
      const input = { cwd: "/repo", image, question: "Why?", modelSelection };

      assert.deepStrictEqual(yield* describeImage(input), { description: "A stack trace." });
      assert.deepStrictEqual(requests[0]?.attachments, [image]);
      assert.include(requests[0]?.prompt, "Why?");

      description = " ";
      const error = yield* Effect.flip(describeImage(input));
      assert.strictEqual(error.detail, "Test returned an empty image description.");
    }),
  );
});
