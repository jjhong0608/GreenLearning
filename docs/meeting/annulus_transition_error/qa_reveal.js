#!/usr/bin/env node

/** Reproducible visual and structural QA for the meeting Reveal.js deck. */

const childProcess = require("child_process");
const fs = require("fs");
const path = require("path");
const { pathToFileURL } = require("url");

class CommandLineConfig {
  constructor(argv) {
    this.values = this.parse(argv);
    this.htmlPath = path.resolve(this.requireValue("html"));
    this.outputDir = path.resolve(this.requireValue("outdir"));
    this.chromePath = this.values.chrome || this.findChrome();
    this.viewports = this.parseViewports(
      this.values.viewports || "1600x900,1280x720",
    );
    this.fragmentSlides = this.parseSlides(
      this.values.slides || "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16",
    );
  }

  parse(argv) {
    const values = {};
    for (let index = 0; index < argv.length; index += 1) {
      const token = argv[index];
      if (!token.startsWith("--")) {
        throw new Error(`Unexpected positional argument: ${token}`);
      }
      const key = token.slice(2);
      const value = argv[index + 1];
      if (!value || value.startsWith("--")) {
        throw new Error(`Missing value for --${key}`);
      }
      values[key] = value;
      index += 1;
    }
    return values;
  }

  requireValue(key) {
    const value = this.values[key];
    if (!value) {
      throw new Error(`Required option --${key} was not provided.`);
    }
    return value;
  }

  parseViewports(raw) {
    return raw.split(",").map((value) => {
      const match = /^(\d+)x(\d+)$/.exec(value.trim());
      if (!match) {
        throw new Error(`Invalid viewport '${value}'. Expected WIDTHxHEIGHT.`);
      }
      return { width: Number(match[1]), height: Number(match[2]) };
    });
  }

  parseSlides(raw) {
    const slides = raw.split(",").map((value) => Number(value.trim()));
    if (slides.some((value) => !Number.isInteger(value) || value < 1)) {
      throw new Error("--slides must be a comma-separated list of positive integers.");
    }
    return slides;
  }

  findChrome() {
    const explicit = process.env.PUPPETEER_EXECUTABLE_PATH;
    if (explicit && fs.existsSync(explicit)) {
      return explicit;
    }
    const searchRoot = path.join(
      process.env.HOME || "",
      ".local",
      "share",
      "decktape-browsers",
      "chrome",
    );
    if (fs.existsSync(searchRoot)) {
      const found = childProcess
        .execFileSync(
          "find",
          [searchRoot, "-type", "f", "-path", "*/chrome-linux64/chrome", "-print", "-quit"],
          { encoding: "utf8" },
        )
        .trim();
      if (found) {
        return found;
      }
    }
    throw new Error("Chrome was not found. Pass --chrome /path/to/chrome.");
  }
}

class PuppeteerLoader {
  static load() {
    try {
      return require("puppeteer");
    } catch (error) {
      const globalRoot = childProcess
        .execFileSync("npm", ["root", "-g"], { encoding: "utf8" })
        .trim();
      const decktapePuppeteer = path.join(
        globalRoot,
        "decktape",
        "node_modules",
        "puppeteer",
      );
      if (!fs.existsSync(decktapePuppeteer)) {
        throw error;
      }
      return require(decktapePuppeteer);
    }
  }
}

class RevealDeckQa {
  constructor(config) {
    this.config = config;
    this.puppeteer = PuppeteerLoader.load();
    this.failures = [];
    this.externalRequests = new Set();
  }

  async run() {
    fs.mkdirSync(this.config.outputDir, { recursive: true });
    const browser = await this.puppeteer.launch({
      executablePath: this.config.chromePath,
      headless: true,
      args: [
        "--allow-file-access-from-files",
        "--enable-unsafe-swiftshader",
        "--no-sandbox",
      ],
    });

    try {
      const report = { html: this.config.htmlPath, viewports: [] };
      for (const viewport of this.config.viewports) {
        report.viewports.push(await this.inspectViewport(browser, viewport));
      }
      const reportPath = path.join(this.config.outputDir, "qa_report.json");
      fs.writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
      console.log(`QA report: ${reportPath}`);
      if (this.failures.length > 0) {
        throw new Error(this.failures.join("\n"));
      }
      console.log(
        "Reveal.js QA passed: no overflow, layout overlap, or external network dependency.",
      );
    } finally {
      await browser.close();
    }
  }

  async inspectCurrentSlide(page) {
    return page.evaluate(() => {
      const current = Reveal.getCurrentSlide();
      const section = current.getBoundingClientRect();
      const tolerance = 2;
      const isVisible = (node) => {
        const style = window.getComputedStyle(node);
        const rect = node.getBoundingClientRect();
        return (
          style.display !== "none" &&
          style.visibility !== "hidden" &&
          Number(style.opacity) > 0.01 &&
          rect.width > 0 &&
          rect.height > 0
        );
      };
      const describe = (node) => ({
        tag: node.tagName,
        className: node.className,
        text: (node.textContent || "").trim().slice(0, 80),
      });
      const overflowNodes = Array.from(current.querySelectorAll("*"))
        .filter(isVisible)
        .filter((node) => {
          const rect = node.getBoundingClientRect();
          return (
            rect.left < section.left - tolerance ||
            rect.right > section.right + tolerance ||
            rect.top < section.top - tolerance ||
            rect.bottom > section.bottom + tolerance
          );
        })
        .slice(0, 12)
        .map(describe);

      const groups = [
        Array.from(current.children).filter(
          (node) => !["ASIDE", "SCRIPT", "STYLE"].includes(node.tagName),
        ),
        ...[
          ".geometry-strip > .slice-card",
          ".pipeline > .formula-card",
          ".expanded-equations > div",
          ".timeline > .timeline-card",
          ".method-grid > .method-card",
          ".result-grid > *",
          ".result-side > *",
        ].map((selector) => Array.from(current.querySelectorAll(selector))),
      ];
      const overlapPairs = [];
      for (const group of groups) {
        const nodes = group.filter(isVisible);
        for (let leftIndex = 0; leftIndex < nodes.length; leftIndex += 1) {
          const left = nodes[leftIndex];
          const leftRect = left.getBoundingClientRect();
          for (
            let rightIndex = leftIndex + 1;
            rightIndex < nodes.length;
            rightIndex += 1
          ) {
            const right = nodes[rightIndex];
            const rightRect = right.getBoundingClientRect();
            const overlapWidth =
              Math.min(leftRect.right, rightRect.right) -
              Math.max(leftRect.left, rightRect.left);
            const overlapHeight =
              Math.min(leftRect.bottom, rightRect.bottom) -
              Math.max(leftRect.top, rightRect.top);
            if (overlapWidth > tolerance && overlapHeight > tolerance) {
              overlapPairs.push({
                left: describe(left),
                right: describe(right),
                overlapWidth,
                overlapHeight,
              });
            }
          }
        }
      }
      return {
        scrollOverflow:
          current.scrollWidth > current.clientWidth + tolerance ||
          current.scrollHeight > current.clientHeight + tolerance,
        overflowNodes,
        overlapPairs: overlapPairs.slice(0, 12),
      };
    });
  }

  async inspectViewport(browser, viewport) {
    const page = await browser.newPage();
    await page.setViewport(viewport);
    const pageErrors = [];
    page.on("pageerror", (error) => pageErrors.push(String(error)));
    page.on("request", (request) => {
      const url = request.url();
      if (/^https?:\/\//.test(url)) {
        this.externalRequests.add(url);
      }
    });

    const fileUrl = pathToFileURL(this.config.htmlPath).href;
    await page.goto(fileUrl, { waitUntil: "networkidle0" });
    await page.waitForFunction(
      () => typeof Reveal !== "undefined" && Reveal.isReady(),
    );
    await page.evaluate(() => Reveal.configure({ transition: "none" }));

    const slides = await page.evaluate(() =>
      Array.from(document.querySelectorAll(".reveal .slides > section")).map(
        (slide, index) => ({
          index,
          id: slide.id || `slide-${index + 1}`,
          maxFragmentIndex: Math.max(
            -1,
            ...Array.from(slide.querySelectorAll(".fragment")).map((fragment) =>
              Number(fragment.dataset.fragmentIndex || 0),
            ),
          ),
        }),
      ),
    );

    if (slides.length !== 18) {
      this.failures.push(
        `${viewport.width}x${viewport.height}: expected 18 slides, found ${slides.length}.`,
      );
    }

    const overflow = [];
    const overlap = [];
    const viewportKey = `${viewport.width}x${viewport.height}`;
    const finalDir = path.join(path.dirname(this.config.outputDir), viewportKey);
    fs.mkdirSync(finalDir, { recursive: true });
    for (const slide of slides) {
      await page.evaluate(
        ({ index, fragment }) => Reveal.slide(index, 0, fragment),
        { index: slide.index, fragment: slide.maxFragmentIndex },
      );
      await new Promise((resolve) => setTimeout(resolve, 180));
      const result = await this.inspectCurrentSlide(page);
      if (result.scrollOverflow || result.overflowNodes.length > 0) {
        overflow.push({
          slide: slide.index + 1,
          id: slide.id,
          ...result,
        });
      }
      if (result.overlapPairs.length > 0) {
        overlap.push({
          slide: slide.index + 1,
          id: slide.id,
          overlapPairs: result.overlapPairs,
        });
      }
      const screenshotPrefix = `annulus_transition_final_${viewport.width}`;
      const screenshotName = `${screenshotPrefix}_${slide.index + 1}_${viewportKey}.png`;
      await page.screenshot({ path: path.join(finalDir, screenshotName) });
    }

    if (overflow.length > 0) {
      this.failures.push(`${viewportKey}: ${overflow.length} slide(s) overflow.`);
    }
    if (overlap.length > 0) {
      this.failures.push(`${viewportKey}: ${overlap.length} slide(s) overlap.`);
    }
    if (pageErrors.length > 0) {
      this.failures.push(`${viewportKey}: page errors: ${pageErrors.join(" | ")}`);
    }
    if (this.externalRequests.size > 0) {
      this.failures.push(
        `${viewportKey}: external requests: ${Array.from(this.externalRequests).join(", ")}`,
      );
    }

    const fragmentOverflow = [];
    const fragmentOverlap = [];
    let fragmentStatesChecked = 0;
    const fragmentDir =
      viewport.width === 1600 && viewport.height === 900
        ? path.join(this.config.outputDir, "fragments")
        : path.join(this.config.outputDir, "fragments", viewportKey);
    fs.mkdirSync(fragmentDir, { recursive: true });
    for (const slideNumber of this.config.fragmentSlides) {
      const slide = slides[slideNumber - 1];
      if (!slide) {
        throw new Error(`Fragment QA requested nonexistent slide ${slideNumber}.`);
      }
      for (let fragment = -1; fragment <= slide.maxFragmentIndex; fragment += 1) {
        await page.evaluate(
          ({ index, fragmentIndex }) => Reveal.slide(index, 0, fragmentIndex),
          { index: slide.index, fragmentIndex: fragment },
        );
        await new Promise((resolve) => setTimeout(resolve, 220));
        fragmentStatesChecked += 1;
        const result = await this.inspectCurrentSlide(page);
        if (result.scrollOverflow || result.overflowNodes.length > 0) {
          fragmentOverflow.push({
            slide: slideNumber,
            state: fragment + 1,
            id: slide.id,
            ...result,
          });
        }
        if (result.overlapPairs.length > 0) {
          fragmentOverlap.push({
            slide: slideNumber,
            state: fragment + 1,
            id: slide.id,
            overlapPairs: result.overlapPairs,
          });
        }
        const state = fragment + 1;
        const filename = `slide_${String(slideNumber).padStart(2, "0")}_state_${String(state).padStart(2, "0")}.png`;
        await page.screenshot({ path: path.join(fragmentDir, filename) });
      }
    }
    if (fragmentOverflow.length > 0) {
      this.failures.push(
        `${viewportKey}: ${fragmentOverflow.length} intermediate fragment state(s) overflow.`,
      );
    }
    if (fragmentOverlap.length > 0) {
      this.failures.push(
        `${viewportKey}: ${fragmentOverlap.length} intermediate fragment state(s) overlap.`,
      );
    }

    await page.close();
    return {
      viewport: viewportKey,
      slideCount: slides.length,
      overflow,
      overlap,
      fragmentStatesChecked,
      fragmentOverflow,
      fragmentOverlap,
      pageErrors,
      externalRequests: Array.from(this.externalRequests),
    };
  }
}

async function main() {
  const config = new CommandLineConfig(process.argv.slice(2));
  const qa = new RevealDeckQa(config);
  await qa.run();
}

main().catch((error) => {
  console.error(error.stack || String(error));
  process.exit(1);
});
