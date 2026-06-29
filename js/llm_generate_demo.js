/*
Run: node llm_generate_demo.js "Hello AI"
*/

const { generateText } = require("./llm_toy_transformer");

const prompt = process.argv[2] ?? "Hello AI";
console.log(generateText(prompt, 80));

