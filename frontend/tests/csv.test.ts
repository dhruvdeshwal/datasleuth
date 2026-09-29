import { toCsv } from "../src/lib/csv";

describe("toCsv", () => {
  it("quotes commas, quotes and newlines", () => {
    expect(toCsv(["a", "b"], [["x,y", 'he said "hi"'], ["line\nbreak", null]])).toBe(
      'a,b\r\n"x,y","he said ""hi"""\r\n"line\nbreak",',
    );
  });

  it("neutralises spreadsheet formulas but keeps negative numbers", () => {
    const csv = toCsv(["v"], [["=SUM(A1)"], ["-5"], [-3], ["@cmd"]]);
    expect(csv).toBe("v\r\n'=SUM(A1)\r\n-5\r\n-3\r\n'@cmd");
  });
});
