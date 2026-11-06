// Render report.html to ../WhisperLah-Technical-Report.pdf with headless Chromium.
//   python docs/technical-report/figures.py && node docs/technical-report/build_pdf.js
const path = require("path");
const { chromium } = require("playwright");
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto("file://" + path.join(__dirname, "report.html"), { waitUntil: "networkidle" });
  await page.pdf({
    path: path.join(__dirname, "..", "WhisperLah-Technical-Report.pdf"),
    format: "A4",
    printBackground: true,
    displayHeaderFooter: true,
    headerTemplate: "<span></span>",
    footerTemplate: '<div style="width:100%;font-family:Inter,sans-serif;font-size:7pt;color:#8A857C;padding:0 19mm;display:flex;justify-content:space-between"><span>WhisperLah technical report v1.0 · Lepaq · November 2026</span><span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>',
    margin: { top: "18mm", bottom: "18mm", left: "19mm", right: "19mm" },
  });
  await browser.close();
})();
