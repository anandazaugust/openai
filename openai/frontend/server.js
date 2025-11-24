const express = require("express");
const path = require("path");
const fetch = require("node-fetch");

const app = express();
const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8080";

app.use(express.static(__dirname));
app.use(express.json());

// Forward /chat
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

// Forward /history
app.get("/history", async (req, res) => {
  const chatId = req.query.chat_id;

  try {
    const response = await fetch(`${BACKEND_URL}/history?chat_id=${chatId}`);
    const data = await response.json();
    res.json(data);
  } catch (err) {
    console.error("Error loading history:", err);
    res.status(500).json({ error: "Failed to load history" });
  }
});

// Forward /list_chats
app.get("/list_chats", async (req, res) => {
  try {
    const response = await fetch(`${BACKEND_URL}/list_chats`);
    const data = await response.json();
    res.json(data);
  } catch (err) {
    console.error("Error loading chat list:", err);
    res.status(500).json({ error: "Failed to load chat list" });
  }
});

app.listen(8080, () => {
  console.log("MyOwnGPT running at http://localhost:8080");
  console.log(`Using backend: ${BACKEND_URL}`);
});
