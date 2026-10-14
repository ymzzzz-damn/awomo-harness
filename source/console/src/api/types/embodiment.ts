export interface EmbodimentPackageInfo {
  key: string;
  package_id: string;
  path: string;
  name: string;
  version: string;
  description: string;
  enabled: boolean;
  bindings: Record<string, string>;
  kind: string;
  tool_count: number;
  path_exists: boolean;
  safety_profile: string;
  hardware_roles: string[];
}

export interface EmbodimentCreateRequest {
  package_key: string;
  package: {
    package_id: string;
    path: string;
    description?: string;
    enabled?: boolean;
    bindings?: Record<string, string>;
  };
}

export interface EmbodimentUpdateRequest {
  package_id?: string;
  path?: string;
  description?: string;
  enabled?: boolean;
  bindings?: Record<string, string>;
}

export interface EmbodimentToolInfo {
  name: string;
  description: string;
  method: string;
  path: string;
  safety_class: "safe" | "confirmation" | "supervisor" | "forbidden";
  input_schema: Record<string, unknown>;
  node_id: string;
  node_name: string;
  latency_estimate_ms?: number;
}

export interface HardwareRequirement {
  role: string;
  description: string;
  device_type: string;
  hint: string;
  bound_endpoint: string;
  default_port?: number;
}

export interface EmbodimentDiscoveryEntry {
  package_id: string;
  name: string;
  path: string;
  version: string;
  kind: string;
  description: string;
  hardware_roles: string[];
  already_configured: boolean;
  config_key?: string;
  hardware_detected: boolean;
  probe_info: Record<string, unknown>;
}
