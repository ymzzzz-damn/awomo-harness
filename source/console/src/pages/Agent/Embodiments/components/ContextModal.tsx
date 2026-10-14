import React, { useState, useCallback, useEffect } from "react";
import { Modal, Button } from "@agentscope-ai/design";
import { Spin } from "antd";
import { useTranslation } from "react-i18next";
import api from "../../../../api";
import styles from "../index.module.less";

interface ContextModalProps {
  open: boolean;
  packageKey: string;
  packageName: string;
  onClose: () => void;
}

export const ContextModal = React.memo(function ContextModal({
  open,
  packageKey,
  packageName,
  onClose,
}: ContextModalProps) {
  const { t } = useTranslation();
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    if (!open) return;
    setLoading(true);
    try {
      const data = await api.getEmbodimentContext(packageKey);
      setContent(data.content);
    } catch {
      setContent("");
    } finally {
      setLoading(false);
    }
  }, [open, packageKey]);

  useEffect(() => {
    if (open) load();
  }, [open, load]);

  return (
    <Modal
      title={`${packageName} — EMBODIMENT.md`}
      open={open}
      onCancel={onClose}
      footer={
        <div style={{ textAlign: "right" }}>
          <Button onClick={onClose}>{t("common.close")}</Button>
        </div>
      }
      width={800}
    >
      {loading ? (
        <div className={styles.centeredSpin}>
          <Spin />
        </div>
      ) : content ? (
        <pre className={styles.mdViewer}>{content}</pre>
      ) : (
        <p style={{ color: "#999" }}>{t("embodiment.noContext")}</p>
      )}
    </Modal>
  );
});
