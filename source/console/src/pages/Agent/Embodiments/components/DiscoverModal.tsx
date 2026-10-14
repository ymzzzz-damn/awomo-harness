import React, { useState, useCallback, useEffect } from "react";
import {
  Modal,
  Button,
  Input,
  Empty,
  Tag,
  Tooltip,
} from "@agentscope-ai/design";
import { SearchOutlined, PlusOutlined, CheckCircleOutlined } from "@ant-design/icons";
import { Spin } from "antd";
import { useTranslation } from "react-i18next";
import type { EmbodimentDiscoveryEntry } from "../../../../api/types";
import api from "../../../../api";
import styles from "../index.module.less";

interface DiscoverModalProps {
  open: boolean;
  onClose: () => void;
  onAddPackage: (
    key: string,
    pkg: {
      package_id: string;
      path: string;
      description?: string;
      enabled?: boolean;
      bindings?: Record<string, string>;
    },
  ) => Promise<boolean>;
}

const KIND_COLOR: Record<string, string> = {
  robot: "blue",
  sensor: "green",
  embodiment_context: "gold",
};

export const DiscoverModal = React.memo(function DiscoverModal({
  open,
  onClose,
  onAddPackage,
}: DiscoverModalProps) {
  const { t } = useTranslation();
  const [packages, setPackages] = useState<EmbodimentDiscoveryEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState("");
  const [adding, setAdding] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.discoverEmbodiments();
      // Sort: hardware-detected first, then by name
      const sorted = [...data].sort((a, b) => {
        if (a.hardware_detected !== b.hardware_detected) return a.hardware_detected ? -1 : 1;
        return a.name.localeCompare(b.name);
      });
      setPackages(sorted);
    } catch {
      setPackages([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) load();
  }, [open, load]);

  const filtered = packages.filter(
    (p) =>
      !search ||
      p.name.toLowerCase().includes(search.toLowerCase()) ||
      p.package_id.toLowerCase().includes(search.toLowerCase()),
  );

  const handleAdd = async (entry: EmbodimentDiscoveryEntry) => {
    setAdding(entry.package_id);
    const key = entry.package_id.replace(/[^a-z0-9_-]/gi, "_");
    const success = await onAddPackage(key, {
      package_id: entry.package_id,
      path: entry.path,
      description: entry.description,
      enabled: true,
      bindings: {},
    });
    setAdding(null);
    if (success) load();
  };

  return (
    <Modal
      title={t("embodiment.discoverTitle")}
      open={open}
      onCancel={onClose}
      footer={
        <div style={{ textAlign: "right" }}>
          <Button onClick={load} style={{ marginRight: 8 }}>
            {t("embodiment.rescan")}
          </Button>
          <Button onClick={onClose}>{t("common.close")}</Button>
        </div>
      }
      width={760}
    >
      <Input
        prefix={<SearchOutlined />}
        placeholder={t("embodiment.searchPackages")}
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        style={{ marginBottom: 12 }}
        allowClear
      />

      {loading ? (
        <div className={styles.centeredSpin}>
          <Spin />
        </div>
      ) : filtered.length === 0 ? (
        <Empty description={t("embodiment.noPackagesFound")} />
      ) : (
        <div className={styles.discoverList}>
          {filtered.map((entry) => (
            <div
              key={entry.package_id}
              className={`${styles.discoverItem} ${entry.hardware_detected ? styles.discoverItemDetected : ""}`}
            >
              <div className={styles.discoverItemHeader}>
                <div className={styles.discoverItemTitle}>
                  <Tag color={KIND_COLOR[entry.kind] || "default"}>
                    {entry.kind || "unknown"}
                  </Tag>
                  <span className={styles.discoverName}>{entry.name}</span>
                  <span className={styles.discoverVersion}>
                    v{entry.version}
                  </span>
                  {entry.hardware_detected && (
                    <Tag color="success" icon={<CheckCircleOutlined />} style={{ fontSize: 11 }}>
                      {t("embodiment.hardwareDetected")}
                    </Tag>
                  )}
                </div>
                {entry.already_configured ? (
                  <Tag color="default" style={{ flexShrink: 0 }}>
                    {t("embodiment.alreadyAdded")} ({entry.config_key})
                  </Tag>
                ) : (
                  <Button
                    type="primary"
                    size="small"
                    icon={<PlusOutlined />}
                    loading={adding === entry.package_id}
                    onClick={() => handleAdd(entry)}
                  >
                    {t("common.add")}
                  </Button>
                )}
              </div>

              <p className={styles.discoverDesc}>{entry.description || "-"}</p>

              <div className={styles.discoverMeta}>
                <Tooltip title={entry.path}>
                  <code className={styles.discoverPath}>{entry.path}</code>
                </Tooltip>
                {entry.hardware_roles.length > 0 && (
                  <div className={styles.discoverRoles}>
                    {entry.hardware_roles.map((r) => (
                      <Tag key={r} color="purple" style={{ fontSize: 11 }}>
                        @{r}
                      </Tag>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </Modal>
  );
});

