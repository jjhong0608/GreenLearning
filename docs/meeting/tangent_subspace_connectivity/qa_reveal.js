#!/usr/bin/env node

/** Browser QA for every final and intermediate Reveal.js fragment state. */

const childProcess = require("child_process");
const fs = require("fs");
const path = require("path");
const { pathToFileURL } = require("url");

const expectedSlideCount = 47;
const defaultViewports = "1600x900,1280x720";

class CommandLineConfig {
  constructor(argv) {
    this.values = this.parse(argv);
    this.htmlPath = path.resolve(this.require("html"));
    this.outputDir = path.resolve(this.require("outdir"));
    this.chromePath = this.values.chrome || this.findChrome();
    this.viewports = this.parseViewports(this.values.viewports || defaultViewports);
    this.slideNumbers = this.parseSlides(
      this.values.slides || Array.from({ length: expectedSlideCount }, (_, index) => index + 1).join(","),
    );
  }

  parse(argv) {
    const values = {};
    for (let index = 0; index < argv.length; index += 1) {
      const token = argv[index];
      if (!token.startsWith("--")) {
        throw new Error(`Unexpected positional argument: ${token}`);
      }
      const value = argv[index + 1];
      if (!value || value.startsWith("--")) {
        throw new Error(`Missing value for ${token}`);
      }
      values[token.slice(2)] = value;
      index += 1;
    }
    return values;
  }

  require(key) {
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
        throw new Error(`Invalid viewport '${value}'.`);
      }
      return { width: Number(match[1]), height: Number(match[2]) };
    });
  }

  parseSlides(raw) {
    const slides = raw.split(",").map((value) => Number(value.trim()));
    if (slides.some((value) => !Number.isInteger(value) || value < 1 || value > expectedSlideCount)) {
      throw new Error(`Slides must be integers from 1 to ${expectedSlideCount}.`);
    }
    return slides;
  }

  findChrome() {
    const explicit = process.env.PUPPETEER_EXECUTABLE_PATH;
    if (explicit && fs.existsSync(explicit)) {
      return explicit;
    }
    const searchRoot = path.join(process.env.HOME || "", ".local", "share", "decktape-browsers", "chrome");
    if (fs.existsSync(searchRoot)) {
      const found = childProcess.execFileSync(
        "find",
        [searchRoot, "-type", "f", "-path", "*/chrome-linux64/chrome", "-print", "-quit"],
        { encoding: "utf8" },
      ).trim();
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
      const globalRoot = childProcess.execFileSync("npm", ["root", "-g"], { encoding: "utf8" }).trim();
      const decktapePuppeteer = path.join(globalRoot, "decktape", "node_modules", "puppeteer");
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
    this.pageErrors = [];
  }

  async run() {
    fs.mkdirSync(this.config.outputDir, { recursive: true });
    const browser = await this.puppeteer.launch({
      executablePath: this.config.chromePath,
      headless: true,
      args: ["--allow-file-access-from-files", "--enable-unsafe-swiftshader", "--no-sandbox"],
    });
    const report = { html: this.config.htmlPath, expectedSlideCount, viewports: [] };
    try {
      for (const viewport of this.config.viewports) {
        report.viewports.push(await this.inspectViewport(browser, viewport));
      }
      report.externalRequests = Array.from(this.externalRequests).sort();
      report.pageErrors = this.pageErrors;
      fs.writeFileSync(
        path.join(this.config.outputDir, "qa_report.json"),
        `${JSON.stringify(report, null, 2)}\n`,
      );
      if (this.externalRequests.size > 0) {
        this.failures.push(`External requests: ${Array.from(this.externalRequests).join(", ")}`);
      }
      if (this.pageErrors.length > 0) {
        this.failures.push(`Page errors: ${this.pageErrors.join(" | ")}`);
      }
      if (this.failures.length > 0) {
        throw new Error(this.failures.join("\n"));
      }
      console.log("Reveal.js QA passed for all requested fragment states and viewports.");
    } finally {
      await browser.close();
    }
  }

  async inspectViewport(browser, viewport) {
    const page = await browser.newPage();
    await page.setViewport(viewport);
    page.on("request", (request) => {
      if (/^https?:/.test(request.url())) {
        this.externalRequests.add(request.url());
      }
    });
    page.on("pageerror", (error) => this.pageErrors.push(String(error)));
    await page.goto(pathToFileURL(this.config.htmlPath).href, { waitUntil: "networkidle0" });
    await page.waitForFunction(() => window.Reveal && Reveal.isReady());
    await page.evaluate(() => Reveal.configure({
      transition: "none",
      backgroundTransition: "none",
    }));
    const slideCount = await page.evaluate(() => Reveal.getSlides().length);
    if (slideCount !== expectedSlideCount) {
      this.failures.push(`${viewport.width}x${viewport.height}: expected ${expectedSlideCount} slides, found ${slideCount}`);
    }

    const viewportDir = path.join(this.config.outputDir, `${viewport.width}x${viewport.height}`);
    fs.mkdirSync(viewportDir, { recursive: true });
    const states = [];
    for (const slideNumber of this.config.slideNumbers) {
      const horizontalIndex = slideNumber - 1;
      await page.evaluate((index) => Reveal.slide(index, 0, -1), horizontalIndex);
      await this.waitForFrames(page);
      const maxFragment = await page.evaluate(() => {
        const fragments = Array.from(Reveal.getCurrentSlide().querySelectorAll(".fragment"));
        const indices = fragments.map((node, index) => {
          const raw = node.getAttribute("data-fragment-index");
          return raw === null ? index : Number(raw);
        });
        return indices.length === 0 ? -1 : Math.max(...indices);
      });
      for (let fragmentIndex = -1; fragmentIndex <= maxFragment; fragmentIndex += 1) {
        await page.evaluate(
          ({ index, fragment }) => Reveal.slide(index, 0, fragment),
          { index: horizontalIndex, fragment: fragmentIndex },
        );
        await this.waitForFrames(page);
        const inspection = await this.inspectCurrentSlide(page);
        states.push({ slideNumber, fragmentIndex, ...inspection });
        if (inspection.failures.length > 0) {
          this.failures.push(
            `${viewport.width}x${viewport.height} slide ${slideNumber} fragment ${fragmentIndex}: ${inspection.failures.join("; ")}`,
          );
        }
      }
      await page.evaluate((index) => Reveal.slide(index, 0, Number.MAX_SAFE_INTEGER), horizontalIndex);
      await this.waitForFrames(page);
      await page.screenshot({
        path: path.join(viewportDir, `slide-${String(slideNumber).padStart(2, "0")}.png`),
        fullPage: false,
      });
    }
    await page.close();
    return { viewport, states };
  }

  async waitForFrames(page) {
    await new Promise((resolve) => setTimeout(resolve, 90));
    await page.evaluate(async () => {
      const frames = Array.from(Reveal.getCurrentSlide().querySelectorAll("iframe"));
      await Promise.all(frames.map((frame) => {
        if (frame.contentDocument && frame.contentDocument.readyState === "complete") {
          return Promise.resolve();
        }
        return new Promise((resolve) => {
          frame.addEventListener("load", resolve, { once: true });
          setTimeout(resolve, 1500);
        });
      }));
      await Promise.all(frames.map(async (frame) => {
        const frameWindow = frame.contentWindow;
        const graph = frame.contentDocument?.querySelector(".plotly-graph-div");
        if (!frameWindow) {
          return;
        }
        const staticImages = Array.from(
          frame.contentDocument?.querySelectorAll(
            '[data-asset-kind="static-mesh-grid"] img',
          ) || [],
        );
        await Promise.all(staticImages.map((node) => {
          if (node.complete) {
            return Promise.resolve();
          }
          return new Promise((resolve) => {
            node.addEventListener("load", resolve, { once: true });
            node.addEventListener("error", resolve, { once: true });
            frameWindow.setTimeout(resolve, 1500);
          });
        }));
        if (!graph) {
          return;
        }
        for (let attempt = 0; attempt < 90; attempt += 1) {
          const hasPlot = graph.classList.contains("js-plotly-plot");
          const hasDrawable = Boolean(
            graph.querySelector("canvas, .main-svg, .svg-container"),
          );
          if (hasPlot && hasDrawable) {
            break;
          }
          await new Promise((resolve) => frameWindow.setTimeout(resolve, 20));
        }
        await new Promise((resolve) => frameWindow.requestAnimationFrame(
          () => frameWindow.requestAnimationFrame(resolve),
        ));
      }));
    });
    if (await page.$(".present iframe")) {
      await new Promise((resolve) => setTimeout(resolve, 350));
    }
  }

  async inspectCurrentSlide(page) {
    return page.evaluate(() => {
      const slide = Reveal.getCurrentSlide();
      const slideRect = slide.getBoundingClientRect();
      const tolerance = 3;
      const failures = [];
      const visible = (node) => {
        const style = getComputedStyle(node);
        const rect = node.getBoundingClientRect();
        return style.display !== "none" && style.visibility !== "hidden" && Number(style.opacity) > 0.01 && rect.width > 0 && rect.height > 0;
      };
      const describe = (node) => `${node.tagName}.${String(node.className).replace(/\s+/g, ".")}`;
      if (slide.scrollWidth > slide.clientWidth + tolerance || slide.scrollHeight > slide.clientHeight + tolerance) {
        failures.push(`slide scroll overflow ${slide.scrollWidth}x${slide.scrollHeight} > ${slide.clientWidth}x${slide.clientHeight}`);
      }
      const checked = Array.from(slide.querySelectorAll(
        ".formula-card, .concept-card, .operator-card, .proof-card, .step-card, .mgs-card, .graph-rule-card, .layer-card, .synthesis-card, .math.display, iframe, table",
      )).filter(visible);
      for (const node of checked) {
        const rect = node.getBoundingClientRect();
        if (
          rect.left < slideRect.left - tolerance ||
          rect.right > slideRect.right + tolerance ||
          rect.top < slideRect.top - tolerance ||
          rect.bottom > slideRect.bottom + tolerance
        ) {
          failures.push(`${describe(node)} outside slide bounds`);
        }
      }
      const groupSelectors = [
        ".compare-grid", ".operator-grid", ".proof-grid", ".global-field-grid", ".pipeline",
        ".direction-flow", ".step-strip", ".recurrence-grid", ".mgs-grid", ".graph-rule-grid",
        ".three-layer-grid", ".final-synthesis", ".metric-strip", ".dual-figure-grid",
      ];
      for (const selector of groupSelectors) {
        for (const group of slide.querySelectorAll(selector)) {
          const children = Array.from(group.children).filter(visible);
          for (let left = 0; left < children.length; left += 1) {
            const a = children[left].getBoundingClientRect();
            for (let right = left + 1; right < children.length; right += 1) {
              const b = children[right].getBoundingClientRect();
              const overlapWidth = Math.min(a.right, b.right) - Math.max(a.left, b.left);
              const overlapHeight = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
              if (overlapWidth > tolerance && overlapHeight > tolerance) {
                failures.push(`${selector} children overlap`);
              }
            }
          }
        }
      }
      for (const frame of slide.querySelectorAll("iframe")) {
        if (!visible(frame)) continue;
        if (!frame.contentDocument || frame.contentDocument.readyState !== "complete") {
          failures.push(`iframe not loaded: ${frame.getAttribute("src")}`);
          continue;
        }
        const plotlyGraph = frame.contentDocument.querySelector(".plotly-graph-div");
        const staticGrid = frame.contentDocument.querySelector(
          '[data-asset-kind="static-mesh-grid"]',
        );
        if (!plotlyGraph && !staticGrid) {
          failures.push(`iframe has no recognized figure: ${frame.getAttribute("src")}`);
          continue;
        }
        if (staticGrid) {
          const images = Array.from(staticGrid.querySelectorAll("img"));
          if (images.length === 0) {
            failures.push(`static mesh grid has no images: ${frame.getAttribute("src")}`);
          } else if (images.some((node) => !node.complete || node.naturalWidth === 0 || node.naturalHeight === 0)) {
            failures.push(`static mesh image failed to load: ${frame.getAttribute("src")}`);
          }
        }
      }
      return { failures };
    });
  }
}

new RevealDeckQa(new CommandLineConfig(process.argv.slice(2))).run().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
