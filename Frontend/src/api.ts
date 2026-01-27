import axios from "axios";

export type GenerateParams = {
  model: "gpt2" | "lstm";
  prompt: string;
  max_length: number;
  temperature: number;
  top_k: number;
  top_p: number;
  repetition_penalty: number;
  num_beams: number;
  do_sample: boolean;
};

/**
 * Send generation request to backend at POST /api/generate
 * Returns { generated_text: string }
 */
export async function generatePoem(
  params: GenerateParams,
): Promise<{ generated_text: string }> {
  const res = await axios.post("/api/generate", params, {
    headers: { "Content-Type": "application/json" },
    timeout: 120000,
  });
  return res.data;
}
