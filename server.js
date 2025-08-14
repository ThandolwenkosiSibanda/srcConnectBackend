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

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

// Core function to create embeddings and upsert into Supabase
async function createEmbedding(id, text) {
  try {
    // 1. Generate embedding
    const response = await openai.embeddings.create({
      model: "text-embedding-3-large",
      input: text,
    });
    const embedding = response.data[0].embedding;
    console.log("Embedding length:", embedding.length);

    // 2. Upsert into Supabase
    const { data, error } = await supabase.from("complaints").upsert(
      {
        id,
        embedding, // store vector in the `embedding` column
      },
      { onConflict: ["id"] }
    );

    if (error) {
      console.error("Supabase upsert error:", error);
    } else {
      console.log("Document upserted:", data);
    }

    return embedding;
  } catch (error) {
    console.error("Error creating embedding:", error);
    throw error; // re-throw to handle in route
  }
}

// API endpoint
app.post("/api/embeddings", async (req, res) => {
  const { data, id } = req.body;

  if (!data || typeof data !== "string") {
    return res
      .status(400)
      .json({ error: "Invalid input: 'data' must be a non-empty string." });
  }

  try {
    const embedding = await createEmbedding(id || null, data);
    res.status(201).json({ embedding });
  } catch (error) {
    res.status(500).json({ error: error.message || "Internal server error" });
  }
});

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
