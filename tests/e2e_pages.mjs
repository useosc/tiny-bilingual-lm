// End-to-end check of the deployed GitHub Pages demo with headless Chrome.
import puppeteer from "puppeteer-core";

const URL = process.argv[2] || "https://useosc.github.io/tiny-bilingual-lm/";
const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: ["--no-sandbox", "--disable-dev-shm-usage"],
});
const page = await browser.newPage();
page.on("console", (m) => console.log("[console]", m.type(), m.text().slice(0, 200)));
page.on("pageerror", (e) => console.log("[pageerror]", String(e).slice(0, 300)));

await page.goto(URL, { waitUntil: "domcontentloaded", timeout: 60000 });
await page.waitForFunction(
  () => document.getElementById("status")?.textContent?.includes("就绪"),
  { timeout: 240000 }
);
console.log("status:", await page.$eval("#status", (e) => e.textContent));

await page.type("#input", process.argv[3] || "人工智能是");
await page.click("#send");
await page.waitForFunction(
  () => document.querySelectorAll(".msg.bot").length >= 2 &&
        document.querySelectorAll(".msg.bot")[1].textContent.length > 4,
  { timeout: 120000 }
);
const out = await page.$$eval(".msg.bot", (els) => els[els.length - 1].textContent);
console.log("OUTPUT:", out.slice(0, 300));
await browser.close();
