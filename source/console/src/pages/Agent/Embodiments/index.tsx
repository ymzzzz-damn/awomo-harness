import { useState, useCallback } from "react";
import { Button, Empty, Modal, Input, Tag } from "@agentscope-ai/design";
import { SearchOutlined, PlusOutlined, DeleteOutlined } from "@ant-design/icons";
import type { EmbodimentPackageInfo } from "../../../api/types";
import {
  EmbodimentCard,
  DiscoverModal,
  AddPackageModal,
} from "./components";
import { useEmbodiments } from "./useEmbodiments";
import { useTranslation } from "react-i18next";
import { PageHeader } from "@/components/PageHeader";
import styles from "./index.module.less";

// ── Bindings editor modal ─────────────────────────────────────────────────

interface BindingRow { role: string; endpoint: string }

function BindingsModal({
  open,
  pkg,
  onClose,
  onSave,
}: {
  open: boolean;
  pkg: EmbodimentPackageInfo | null;
  onClose: () => void;
  onSave: (key: string, bindings: Record<string, string>) => Promise<boolean>;
}) {
  const { t } = useTranslation();
  const [rows, setRows] = useState<BindingRow[]>([]);
  const [saving, setSaving] = useState(false);

  const initRows = useCallback((p: EmbodimentPackageInfo) => {
    const existing = Object.entries(p.bindings).map(([role, endpoint]) => ({ role, endpoint }));
    const bound = new Set(Object.keys(p.bindings));
    const missing = (p.hardware_roles || []).filter((r) => !bound.has(r)).map((r) => ({ role: r, endpoint: "" }));
    setRows([...existing, ...missing]);
  }, []);

  const handleSave = async () => {
    if (!pkg) return;
    const bindingsMap: Record<string, string> = {};
    for (const r of rows) {
      if (r.role.trim() && r.endpoint.trim()) {
        bindingsMap[r.role.trim()] = r.endpoint.trim();
      }
    }
    setSaving(true);
    const ok = await onSave(pkg.key, bindingsMap);
    setSaving(false);
    if (ok) onClose();
  };

  if (!pkg) return null;

  return (
    <Modal
      title={`${pkg.name || pkg.key} — ${t("embodiment.hardwareBindings")}`}
      open={open}
      onCancel={onClose}
      footer={
        <div style={{ textAlign: "right" }}>
          <Button onClick={onClose} style={{ marginRight: 8 }}>{t("common.cancel")}</Button>
          <Button type="primary" loading={saving} onClick={handleSave}>{t("common.save")}</Button>
        </div>
      }
      width={640}
      destroyOnHidden
      afterOpenChange={(v) => { if (v && pkg) initRows(pkg); }}
    >
      <p className={styles.bindingsHint}>{t("embodiment.bindingsModalHint")}</p>
      {pkg.hardware_roles.length > 0 && (
        <div className={styles.hwRolesNeeded}>
          <span className={styles.hwRolesLabel}>{t("embodiment.rolesRequired")}:</span>
          {pkg.hardware_roles.map((r) => (
            <Tag key={r} color="purple">@{r}</Tag>
          ))}
        </div>
      )}
      <div className={styles.bindingRows}>
        {rows.map((b, idx) => (
          <div key={idx} className={styles.bindingRow}>
            <Input
              value={b.role}
              onChange={(e) => setRows((prev) => prev.map((r, i) => i === idx ? { ...r, role: e.target.value } : r))}
              placeholder="role"
              style={{ width: 200, marginRight: 8 }}
            />
            <span className={styles.bindingArrow}>→</span>
            <Input
              value={b.endpoint}
              onChange={(e) => setRows((prev) => prev.map((r, i) => i === idx ? { ...r, endpoint: e.target.value } : r))}
              placeholder="http://192.168.1.100:8080"
              style={{ flex: 1, marginLeft: 8, marginRight: 8 }}
            />
            <Button danger icon={<DeleteOutlined />} onClick={() => setRows((prev) => prev.filter((_, i) => i !== idx))} />
          </div>
        ))}
        <Button icon={<PlusOutlined />} size="small" onClick={() => setRows((prev) => [...prev, { role: "", endpoint: "" }])}>
          {t("embodiment.addBinding")}
        </Button>
      </div>
    </Modal>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────

function EmbodimentsPage() {
  const { t } = useTranslation();
  const { packages, loading, toggleEnabled, deletePackage, createPackage, updatePackage } = useEmbodiments();

  const [discoverOpen, setDiscoverOpen] = useState(false);
  const [addOpen, setAddOpen] = useState(false);
  const [bindingsPkg, setBindingsPkg] = useState<EmbodimentPackageInfo | null>(null);
  const [search, setSearch] = useState("");

  const filtered = packages.filter(
    (p) =>
      !search ||
      p.key.toLowerCase().includes(search.toLowerCase()) ||
      (p.name || "").toLowerCase().includes(search.toLowerCase()) ||
      p.package_id.toLowerCase().includes(search.toLowerCase()),
  );

  const handleSaveBindings = useCallback(
    async (key: string, bindings: Record<string, string>) => {
      return updatePackage(key, { bindings });
    },
    [updatePackage],
  );

  return (
    <div className={styles.embodimentsPage}>
      <PageHeader
        items={[{ title: t("nav.agent") }, { title: t("embodiment.title") }]}
        extra={
          <div style={{ display: "flex", gap: 8 }}>
            <Button
              icon={<SearchOutlined />}
              onClick={() => setDiscoverOpen(true)}
            >
              {t("embodiment.autoDiscover")}
            </Button>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => setAddOpen(true)}
            >
              {t("embodiment.addManually")}
            </Button>
          </div>
        }
      />

      {packages.length > 0 && (
        <div className={styles.searchBar}>
          <Input
            prefix={<SearchOutlined />}
            placeholder={t("embodiment.searchPackages")}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            allowClear
            style={{ maxWidth: 320 }}
          />
        </div>
      )}

      {loading ? (
        <div className={styles.loading}>
          <p>{t("common.loading")}</p>
        </div>
      ) : filtered.length === 0 ? (
        <div className={styles.emptyState}>
          <Empty description={
            packages.length === 0
              ? t("embodiment.emptyState")
              : t("embodiment.noSearchResults")
          } />
          {packages.length === 0 && (
            <div style={{ marginTop: 16, display: "flex", gap: 8, justifyContent: "center" }}>
              <Button onClick={() => setDiscoverOpen(true)} icon={<SearchOutlined />}>
                {t("embodiment.autoDiscover")}
              </Button>
              <Button type="primary" onClick={() => setAddOpen(true)} icon={<PlusOutlined />}>
                {t("embodiment.addManually")}
              </Button>
            </div>
          )}
        </div>
      ) : (
        <div className={styles.embodimentsGrid}>
          {filtered.map((pkg) => (
            <EmbodimentCard
              key={pkg.key}
              pkg={pkg}
              onToggle={toggleEnabled}
              onDelete={deletePackage}
              onEditBindings={(p) => setBindingsPkg(p)}
            />
          ))}
        </div>
      )}

      <DiscoverModal
        open={discoverOpen}
        onClose={() => setDiscoverOpen(false)}
        onAddPackage={createPackage}
      />

      <AddPackageModal
        open={addOpen}
        onClose={() => setAddOpen(false)}
        onAdd={createPackage}
      />

      <BindingsModal
        open={bindingsPkg !== null}
        pkg={bindingsPkg}
        onClose={() => setBindingsPkg(null)}
        onSave={handleSaveBindings}
      />
    </div>
  );
}

export default EmbodimentsPage;
