
import axios from "axios";

const api = axios.create({
  baseURL: "http://127.0.0.1:8000",
  timeout: 120000,
});

export const analyzeMerchantQuestion = async (question) => {
  const response = await api.post("/api/ai/analyze", {
    question,
  });

  return response.data;
};

export const simulateBacklog = async (config) => {
  const response = await api.post("/api/backlog/simulate", config);
  return response.data;
};

export default api;
