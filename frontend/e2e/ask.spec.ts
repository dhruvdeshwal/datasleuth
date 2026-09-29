import { expect, test } from "@playwright/test";
import path from "node:path";

test("upload a CSV, ask a question, see answer, chart and the self-correction trace", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Upload a file to get started")).toBeVisible();

  await page.getByTestId("file-input").setInputFiles(path.join(process.cwd(), "e2e", "sample.csv"));
  await expect(page.getByText("sample", { exact: true })).toBeVisible();
  await expect(page.getByText("currency (symbols stripped)")).toBeVisible();

  await page.getByLabel("Ask a question about your data").fill("Which region had the highest total sales?");
  await page.getByRole("button", { name: "Ask" }).click();

  await expect(page.getByText(/East has the highest total at 300/)).toBeVisible({ timeout: 30_000 });
  await expect(page.locator(".js-plotly-plot")).toBeVisible();
  await expect(page.getByLabel("sql code").first()).toContainText("SUM(amount)");
  await expect(page.getByRole("table").first()).toContainText("east");

  // Regression: long SQL / wide tables must scroll inside their card, never widen the page.
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);

  await page.getByText(/Agent trace/).click();
  await expect(page.getByText("Diagnosis: unknown_column")).toBeVisible();
  await expect(page.getByText(/Similar columns: .*sample\.amount/)).toBeVisible();

  await page.getByRole("button", { name: /switch to dark mode/i }).click();
  await expect(page.locator("html")).toHaveClass(/dark/);
});
