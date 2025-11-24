const express = require("express");
const path = require("path");
const fetch = require("node-fetch");

const app = express();
const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8080";

app.use(express.static(__dirname));
app.use(express.json());

// ---- Proxy for sending chat messages ----
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

// ---- Proxy to load Redis chat history ----
app.get("/history", async (req, res) => {
  const chatId = req.query.chat_id;

  try {
    const response = await fetch(`${BACKEND_URL}/history?chat_id=${chatId}`);
    const data = await response.json();
    res.json(data);
  } catch (err) {
    console.error("Error calling backend:", err);
    res.status(500).json({ error: "Failed to load history" });
  }
});

app.listen(8080, () => {
  console.log("MyOwnGPT running at http://localhost:8080");
  console.log(`Using backend: ${BACKEND_URL}`);
});
