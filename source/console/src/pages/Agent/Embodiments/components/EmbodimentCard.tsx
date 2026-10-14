import { Card, Button, Modal, Tag, Tooltip } from "@agentscope-ai/design";
import type { EmbodimentPackageInfo } from "../../../../api/types";
import { useTranslation } from "react-i18next";
import React, { useState } from "react";
import {
  EyeOutlined,
  EyeInvisibleOutlined,
  ToolOutlined,
  FileTextOutlined,
  ApiOutlined,
} from "@ant-design/icons";
import { ToolsModal } from "./ToolsModal";
import { ContextModal } from "./ContextModal";
import styles from "../index.module.less";

const KIND_COLOR: Record<string, string> = {
  robot: "blue",
  sensor: "green",
  embodiment_context: "gold",
};

interface EmbodimentCardProps {
  pkg: EmbodimentPackageInfo;
  onToggle: (pkg: EmbodimentPackageInfo) => void;
  onDelete: (pkg: EmbodimentPackageInfo) => void;
  onEditBindings: (pkg: EmbodimentPackageInfo) => void;
}

export const EmbodimentCard = React.memo(function EmbodimentCard({
  pkg,
  onToggle,
  onDelete,
  onEditBindings,
}: EmbodimentCardProps) {
  const { t } = useTranslation();
  const [isHovered, setIsHovered] = useState(false);
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [toolsOpen, setToolsOpen] = useState(false);
  const [contextOpen, setContextOpen] = useState(false);

  const displayName = pkg.name || pkg.key;

  return (
    <>
      <Card
        hoverable
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        className={`${styles.embodimentCard} ${
          pkg.enabled ? styles.enabledCard : ""
        } ${isHovered ? styles.hover : styles.normal}`}
      >
        {/* ── Header ── */}
        <div className={styles.cardHeader}>
          <div className={styles.cardTitleRow}>
            <Tooltip title={pkg.key}>
              <span className={styles.embodimentTitle}>{displayName}</span>
            </Tooltip>
            {pkg.kind && (
              <Tag color={KIND_COLOR[pkg.kind] || "default"} style={{ fontSize: 11, marginLeft: 6 }}>
                {pkg.kind}
              </Tag>
            )}
            {!pkg.path_exists && (
              <Tag color="red" style={{ fontSize: 11, marginLeft: 4 }}>
                {t("embodiment.pathMissing")}
              </Tag>
            )}
          </div>
          <div className={styles.statusContainer}>
            <span className={`${styles.statusDot} ${pkg.enabled ? styles.statusDotEnabled : ""}`} />
            <span className={styles.statusText}>
              {pkg.enabled ? t("common.enabled") : t("common.disabled")}
            </span>
          </div>
        </div>

        {/* ── Description ── */}
        <p className={styles.embodimentDescription}>
          {pkg.description || "-"}
        </p>

        {/* ── Meta strip ── */}
        <div className={styles.metaStrip}>
          <span className={styles.metaPill}>v{pkg.version || "?"}</span>
          <span className={styles.metaPill}>
            <ToolOutlined style={{ marginRight: 3 }} />
            {pkg.tool_count} {t("embodiment.tools")}
          </span>
          {pkg.safety_profile && (
            <span className={`${styles.metaPill} ${styles.safetyPill}`}>
              {pkg.safety_profile}
            </span>
          )}
          {pkg.hardware_roles.length > 0 && (
            <Tooltip
              title={pkg.hardware_roles.map((r) => `@${r}`).join(", ")}
            >
              <span className={`${styles.metaPill} ${styles.hwPill}`}>
                <ApiOutlined style={{ marginRight: 3 }} />
                {pkg.hardware_roles.length} {t("embodiment.roles")}
                {Object.keys(pkg.bindings).length > 0 && (
                  <span className={styles.boundBadge}>
                    {Object.keys(pkg.bindings).length}/{pkg.hardware_roles.length}
                  </span>
                )}
              </span>
            </Tooltip>
          )}
        </div>

        {/* ── Bindings preview ── */}
        {Object.keys(pkg.bindings).length > 0 && (
          <div className={styles.bindingPreview}>
            {Object.entries(pkg.bindings)
              .slice(0, 2)
              .map(([role, ep]) => (
                <div key={role} className={styles.bindingLine}>
                  <Tag color="purple" style={{ fontSize: 10 }}>
                    @{role}
                  </Tag>
                  <code className={styles.bindingEndpoint}>{ep}</code>
                </div>
              ))}
            {Object.keys(pkg.bindings).length > 2 && (
              <span className={styles.bindingMore}>
                +{Object.keys(pkg.bindings).length - 2} more
              </span>
            )}
          </div>
        )}

        {/* ── Action buttons ── */}
        <div className={styles.cardFooter}>
          <Button
            size="small"
            icon={<ToolOutlined />}
            onClick={() => setToolsOpen(true)}
            disabled={!pkg.path_exists}
          >
            {t("embodiment.tools")}
          </Button>
          <Button
            size="small"
            icon={<FileTextOutlined />}
            onClick={() => setContextOpen(true)}
            disabled={!pkg.path_exists}
          >
            EMBODIMENT.md
          </Button>
          <Button
            size="small"
            icon={<ApiOutlined />}
            onClick={() => onEditBindings(pkg)}
          >
            {t("embodiment.bindings")}
          </Button>
          <Button
            size="small"
            icon={pkg.enabled ? <EyeInvisibleOutlined /> : <EyeOutlined />}
            onClick={() => onToggle(pkg)}
          >
            {pkg.enabled ? t("common.disable") : t("common.enable")}
          </Button>
          <Button
            size="small"
            danger
            onClick={() => setDeleteModalOpen(true)}
          >
            {t("common.delete")}
          </Button>
        </div>
      </Card>

      <Modal
        title={t("common.confirm")}
        open={deleteModalOpen}
        onOk={() => { setDeleteModalOpen(false); onDelete(pkg); }}
        onCancel={() => setDeleteModalOpen(false)}
        okText={t("common.confirm")}
        cancelText={t("common.cancel")}
        okButtonProps={{ danger: true }}
      >
        <p>{t("embodiment.deleteConfirm")}</p>
      </Modal>

      <ToolsModal
        open={toolsOpen}
        packageKey={pkg.key}
        packageName={displayName}
        onClose={() => setToolsOpen(false)}
      />

      <ContextModal
        open={contextOpen}
        packageKey={pkg.key}
        packageName={displayName}
        onClose={() => setContextOpen(false)}
      />
    </>
  );
});
