import fs from "fs";
import { BPE } from "../web/bpe.js";

const tokJson = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const bpe = new BPE(tokJson);
const cases = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
let bad = 0;
for (const c of cases) {
  const ids = bpe.encode(c.text);
  const ok = JSON.stringify(ids) === JSON.stringify(c.ids);
  if (!ok) bad++;
  console.log(`${ok ? "OK " : "BAD"} ${JSON.stringify(c.text).slice(0, 40)} py=${c.ids.length} js=${ids.length}`);
  if (!ok) console.log("  py", c.ids.slice(0, 20), "\n  js", ids.slice(0, 20));
}
console.log(bad === 0 ? "ALL OK" : `${bad} mismatches`);
