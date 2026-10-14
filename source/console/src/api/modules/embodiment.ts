import { request } from "../request";
import type {
  EmbodimentPackageInfo,
  EmbodimentCreateRequest,
  EmbodimentUpdateRequest,
  EmbodimentToolInfo,
  HardwareRequirement,
  EmbodimentDiscoveryEntry,
} from "../types";

export const embodimentApi = {
  listEmbodiments: () =>
    request<EmbodimentPackageInfo[]>("/embodiments"),

  discoverEmbodiments: () =>
    request<EmbodimentDiscoveryEntry[]>("/embodiments/discover"),

  getEmbodiment: (key: string) =>
    request<EmbodimentPackageInfo>(`/embodiments/${encodeURIComponent(key)}`),

  getEmbodimentTools: (key: string) =>
    request<EmbodimentToolInfo[]>(
      `/embodiments/${encodeURIComponent(key)}/tools`,
    ),

  getEmbodimentContext: (key: string) =>
    request<{ content: string; package_id: string }>(
      `/embodiments/${encodeURIComponent(key)}/context`,
    ),

  getEmbodimentHardware: (key: string) =>
    request<HardwareRequirement[]>(
      `/embodiments/${encodeURIComponent(key)}/hardware`,
    ),

  createEmbodiment: (body: EmbodimentCreateRequest) =>
    request<EmbodimentPackageInfo>("/embodiments", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  updateEmbodiment: (key: string, body: EmbodimentUpdateRequest) =>
    request<EmbodimentPackageInfo>(
      `/embodiments/${encodeURIComponent(key)}`,
      { method: "PUT", body: JSON.stringify(body) },
    ),

  toggleEmbodiment: (key: string) =>
    request<EmbodimentPackageInfo>(
      `/embodiments/${encodeURIComponent(key)}/toggle`,
      { method: "PATCH" },
    ),

  deleteEmbodiment: (key: string) =>
    request<{ message: string }>(
      `/embodiments/${encodeURIComponent(key)}`,
      { method: "DELETE" },
    ),
};
