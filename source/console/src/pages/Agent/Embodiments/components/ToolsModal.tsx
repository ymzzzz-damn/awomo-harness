import React, { useState, useCallback } from "react";
import { Modal, Button, Empty, Tag } from "@agentscope-ai/design";
import { Spin } from "antd";
import { useTranslation } from "react-i18next";
import type { EmbodimentToolInfo } from "../../../../api/types";
import api from "../../../../api";
import styles from "../index.module.less";

const SAFETY_COLOR: Record<string, string> = {
  safe: "green",
  confirmation: "orange",
  supervisor: "volcano",
  forbidden: "red",
};

interface ToolsModalProps {
  open: boolean;
  packageKey: string;
  packageName: string;
  onClose: () => void;
}

export const ToolsModal = React.memo(function ToolsModal({
  open,
  packageKey,
  packageName,
  onClose,
}: ToolsModalProps) {
  const { t } = useTranslation();
  const [tools, setTools] = useState<EmbodimentToolInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadTools = useCallback(async () => {
    if (!open) return;
    setLoading(true);
    setError(null);
    try {
      const data = await api.getEmbodimentTools(packageKey);
      setTools(data);
    } catch (err: any) {
      setError(err?.message || t("embodiment.toolsLoadError"));
    } finally {
      setLoading(false);
    }
  }, [open, packageKey, t]);

  React.useEffect(() => {
    if (open) loadTools();
  }, [open, loadTools]);

  const grouped = tools.reduce<Record<string, EmbodimentToolInfo[]>>(
    (acc, tool) => {
      (acc[tool.node_name] = acc[tool.node_name] || []).push(tool);
      return acc;
    },
    {},
  );

  return (
    <Modal
      title={`${packageName} — ${t("embodiment.tools")}`}
      open={open}
      onCancel={onClose}
      footer={
        <div style={{ textAlign: "right" }}>
          <Button onClick={onClose}>{t("common.close")}</Button>
        </div>
      }
      width={760}
    >
      {loading ? (
        <div className={styles.centeredSpin}>
          <Spin />
        </div>
      ) : error ? (
        <div className={styles.errorText}>{error}</div>
      ) : tools.length === 0 ? (
        <Empty description={t("embodiment.noTools")} />
      ) : (
        <div className={styles.toolsList}>
          {Object.entries(grouped).map(([nodeName, nodeTools]) => (
            <div key={nodeName} className={styles.nodeGroup}>
              <div className={styles.nodeGroupTitle}>{nodeName}</div>
              {nodeTools.map((tool) => (
                <div key={tool.name} className={styles.toolItem}>
                  <div className={styles.toolHeader}>
                    <Tag color={SAFETY_COLOR[tool.safety_class] || "default"}>
                      {tool.safety_class}
                    </Tag>
                    <span className={styles.toolMethod}>{tool.method}</span>
                    <code className={styles.toolPath}>{tool.path}</code>
                    {tool.latency_estimate_ms != null && (
                      <span className={styles.toolLatency}>
                        ~{tool.latency_estimate_ms}ms
                      </span>
                    )}
                  </div>
                  <div className={styles.toolName}>{tool.name}</div>
                  {tool.description && (
                    <p className={styles.toolDescription}>{tool.description}</p>
                  )}
                  {tool.input_schema &&
                    Object.keys(tool.input_schema).length > 0 && (
                      <details className={styles.toolSchema}>
                        <summary>{t("embodiment.inputSchema")}</summary>
                        <pre className={styles.toolSchemaContent}>
                          {JSON.stringify(tool.input_schema, null, 2)}
                        </pre>
                      </details>
                    )}
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </Modal>
  );
});
