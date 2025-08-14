import express from "express";
import { createClient } from "@supabase/supabase-js";
import { Pool } from "pg";
import dotenv from "dotenv";
import cors from "cors";
import OpenAI from "openai";

dotenv.config();

const app = express();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

app.use(
  cors({
    origin: ["https://your-production-site.com", "http://localhost:3000"],
    credentials: true,
  })
);

app.use(express.json()); // parse JSON bodies

const supabaseAdmin = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

// Core function to create embeddings
async function createEmbedding(text) {
  try {
    const response = await openai.embeddings.create({
      model: "text-embedding-3-large",
      input: text,
    });
    const embedding = response.data[0].embedding;
    console.log("Embedding length:", embedding.length);
    return embedding;
  } catch (error) {
    console.error("Error creating embedding:", error);
  }
}

app.post("/api/embeddings", async (req, res) => {
  const { data } = req.body;

  if (!data || typeof data !== "string") {
    return res
      .status(400)
      .json({ error: "Invalid input: 'data' must be a non-empty string." });
  }

  try {
    const embedding = await createEmbedding(data);
    res.status(201).json({ embedding });
  } catch (error) {
    console.error("error", error);
    res.status(500).json({ error: error.message || "Internal server error" });
  }
});

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
