import assert from "node:assert/strict";
import { test } from "node:test";

import { rMultiple } from "./formatters.js";

test("rMultiple shows a dash when the trade has no stop", () => {
  assert.equal(rMultiple(null), "—");
  assert.equal(rMultiple(undefined), "—");
});

test("rMultiple keeps zero and signed values", () => {
  assert.equal(rMultiple(0), "0R");
  assert.equal(rMultiple(-1.21), "-1.21R");
  assert.equal(rMultiple(1), "1R");
});
