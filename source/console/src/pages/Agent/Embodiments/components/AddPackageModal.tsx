import React, { useState, useCallback, useEffect } from "react";
import {
  Modal,
  Button,
  Input,
  Tag,
  Tooltip,
} from "@agentscope-ai/design";
import { PlusOutlined, DeleteOutlined } from "@ant-design/icons";
import { useTranslation } from "react-i18next";
import type { HardwareRequirement } from "../../../../api/types";
import api from "../../../../api";
import styles from "../index.module.less";

interface AddPackageModalProps {
  open: boolean;
  onClose: () => void;
  onAdd: (
    key: string,
    pkg: {
      package_id: string;
      path: string;
      description?: string;
      enabled?: boolean;
      bindings?: Record<string, string>;
    },
  ) => Promise<boolean>;
  initialPath?: string;
  initialPackageId?: string;
}

interface BindingRow {
  role: string;
  endpoint: string;
}

export const AddPackageModal = React.memo(function AddPackageModal({
  open,
  onClose,
  onAdd,
  initialPath = "",
  initialPackageId = "",
}: AddPackageModalProps) {
  const { t } = useTranslation();
  const [packageKey, setPackageKey] = useState("");
  const [path, setPath] = useState(initialPath);
  const [packageId, setPackageId] = useState(initialPackageId);
  const [description, setDescription] = useState("");
  const [enabled, setEnabled] = useState(true);
  const [bindings, setBindings] = useState<BindingRow[]>([]);
  const [hwReqs, setHwReqs] = useState<HardwareRequirement[]>([]);
  const [loadingHw, setLoadingHw] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [pathError, setPathError] = useState("");

  useEffect(() => {
    if (open) {
      setPackageKey(initialPackageId.replace(/[^a-z0-9_-]/gi, "_"));
      setPath(initialPath);
      setPackageId(initialPackageId);
      setDescription("");
      setEnabled(true);
      setBindings([]);
      setHwReqs([]);
      setPathError("");
    }
  }, [open, initialPath, initialPackageId]);

  const handlePathChange = (newPath: string) => {
    setPath(newPath);
    setPathError("");
  };

  const handleReadMeta = useCallback(async () => {
    if (!path.trim()) return;
    setLoadingHw(true);
    try {
      // Create a temp key to read hardware requirements
      // We'll use the discover endpoint or read yaml client-side
      // Instead: read from the path by calling discover and matching
      const discovered = await api.discoverEmbodiments();
      const match = discovered.find((d) => d.path === path.trim() || d.path === path.trim().replace(/\/$/, ""));
      if (match) {
        if (!packageId) setPackageId(match.package_id);
        if (!packageKey) setPackageKey(match.package_id.replace(/[^a-z0-9_-]/gi, "_"));
        if (!description) setDescription(match.description);
        // Set empty binding rows for each required role
        setBindings(match.hardware_roles.map((r) => ({ role: r, endpoint: "" })));
        setHwReqs([]);
      } else {
        setPathError(t("embodiment.pathNotFound"));
      }
    } catch {
      setPathError(t("embodiment.pathReadError"));
    } finally {
      setLoadingHw(false);
    }
  }, [path, packageId, packageKey, description, t]);

  const handleAddBinding = () => {
    setBindings((prev) => [...prev, { role: "", endpoint: "" }]);
  };

  const handleRemoveBinding = (idx: number) => {
    setBindings((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleBindingChange = (idx: number, field: "role" | "endpoint", val: string) => {
    setBindings((prev) =>
      prev.map((b, i) => (i === idx ? { ...b, [field]: val } : b)),
    );
  };

  const handleSubmit = async () => {
    if (!packageKey.trim()) return;
    if (!path.trim()) return;

    const bindingsMap: Record<string, string> = {};
    for (const b of bindings) {
      if (b.role.trim() && b.endpoint.trim()) {
        bindingsMap[b.role.trim()] = b.endpoint.trim();
      }
    }

    setSubmitting(true);
    const success = await onAdd(packageKey.trim(), {
      package_id: packageId.trim() || packageKey.trim(),
      path: path.trim(),
      description,
      enabled,
      bindings: bindingsMap,
    });
    setSubmitting(false);
    if (success) onClose();
  };

  return (
    <Modal
      title={t("embodiment.addManually")}
      open={open}
      onCancel={onClose}
      footer={
        <div style={{ textAlign: "right" }}>
          <Button onClick={onClose} style={{ marginRight: 8 }}>
            {t("common.cancel")}
          </Button>
          <Button
            type="primary"
            loading={submitting}
            onClick={handleSubmit}
            disabled={!packageKey.trim() || !path.trim()}
          >
            {t("common.add")}
          </Button>
        </div>
      }
      width={680}
    >
      <div className={styles.formSection}>
        {/* Path + auto-read */}
        <div className={styles.formRow}>
          <label className={styles.formLabel}>
            {t("embodiment.packagePath")} *
          </label>
          <div style={{ display: "flex", gap: 8 }}>
            <Input
              value={path}
              onChange={(e) => handlePathChange(e.target.value)}
              placeholder="/path/to/package"
              style={{ flex: 1 }}
            />
            <Button onClick={handleReadMeta} loading={loadingHw}>
              {t("embodiment.readMeta")}
            </Button>
          </div>
          {pathError && (
            <span className={styles.fieldError}>{pathError}</span>
          )}
          <span className={styles.fieldHint}>{t("embodiment.pathHint")}</span>
        </div>

        {/* Key */}
        <div className={styles.formRow}>
          <label className={styles.formLabel}>
            {t("embodiment.packageKey")} *
          </label>
          <Input
            value={packageKey}
            onChange={(e) => setPackageKey(e.target.value)}
            placeholder="my-robot"
          />
          <span className={styles.fieldHint}>{t("embodiment.keyHint")}</span>
        </div>

        {/* Package ID */}
        <div className={styles.formRow}>
          <label className={styles.formLabel}>{t("embodiment.packageId")}</label>
          <Input
            value={packageId}
            onChange={(e) => setPackageId(e.target.value)}
            placeholder={t("embodiment.packageIdPlaceholder")}
          />
        </div>

        {/* Description */}
        <div className={styles.formRow}>
          <label className={styles.formLabel}>{t("embodiment.descriptionLabel")}</label>
          <Input.TextArea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            autoSize={{ minRows: 2, maxRows: 4 }}
            placeholder={t("embodiment.descriptionPlaceholder")}
          />
        </div>

        {/* Enabled */}
        <div className={styles.formRow}>
          <label className={styles.formLabel}>{t("common.enabled")}</label>
          <Button
            size="small"
            type={enabled ? "primary" : "default"}
            onClick={() => setEnabled((v) => !v)}
          >
            {enabled ? t("common.enabled") : t("common.disabled")}
          </Button>
        </div>

        {/* Hardware bindings */}
        <div className={styles.formRow}>
          <label className={styles.formLabel}>
            {t("embodiment.hardwareBindings")}
            <Tooltip title={t("embodiment.bindingsHint")}>
              <span className={styles.infoIcon}>?</span>
            </Tooltip>
          </label>

          {hwReqs.length > 0 && (
            <div className={styles.hwReqList}>
              {hwReqs.map((r) => (
                <div key={r.role} className={styles.hwReqItem}>
                  <Tag color="purple">@{r.role}</Tag>
                  <span className={styles.hwReqDesc}>{r.description}</span>
                  {r.default_port && (
                    <span className={styles.hwReqPort}>:{r.default_port}</span>
                  )}
                </div>
              ))}
            </div>
          )}

          <div className={styles.bindingRows}>
            {bindings.map((b, idx) => (
              <div key={idx} className={styles.bindingRow}>
                <Input
                  value={b.role}
                  onChange={(e) => handleBindingChange(idx, "role", e.target.value)}
                  placeholder="role (e.g. robot_local_view)"
                  style={{ width: 200, marginRight: 8 }}
                />
                <Input
                  value={b.endpoint}
                  onChange={(e) => handleBindingChange(idx, "endpoint", e.target.value)}
                  placeholder="http://192.168.1.100:8080"
                  style={{ flex: 1, marginRight: 8 }}
                />
                <Button
                  danger
                  icon={<DeleteOutlined />}
                  onClick={() => handleRemoveBinding(idx)}
                />
              </div>
            ))}
            <Button
              icon={<PlusOutlined />}
              onClick={handleAddBinding}
              size="small"
            >
              {t("embodiment.addBinding")}
            </Button>
          </div>
        </div>
      </div>
    </Modal>
  );
});
