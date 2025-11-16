const express = require("express");
const path = require("path");
const fetch = require("node-fetch");

const app = express();
const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8080";

app.use(express.static(__dirname));
app.use(express.json());

app.post("/chat", async (req, res) => {
  try {
    const response = await fetch(`${BACKEND_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    });

    const data = await response.json();
    res.json(data);
  } catch (err) {
    console.error("Error calling backend:", err);
    res.status(500).json({ error: "Failed to connect to backend" });
  }
});

app.listen(8080, () => {
  console.log(" MyOwnGPT running at http://localhost:8080");
  console.log(`Using backend: ${BACKEND_URL}`);
});
