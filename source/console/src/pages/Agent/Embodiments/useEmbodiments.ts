import { useCallback, useEffect, useState } from "react";
import { useAppMessage } from "../../../hooks/useAppMessage";
import api from "../../../api";
import type { EmbodimentPackageInfo } from "../../../api/types";
import { useTranslation } from "react-i18next";
import { useAgentStore } from "../../../stores/agentStore";

export function useEmbodiments() {
  const { t } = useTranslation();
  const { selectedAgent } = useAgentStore();
  const [packages, setPackages] = useState<EmbodimentPackageInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const { message } = useAppMessage();

  const loadPackages = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.listEmbodiments();
      setPackages(data);
    } catch (error) {
      console.error("Failed to load embodiments:", error);
      message.error(t("embodiment.loadError"));
    } finally {
      setLoading(false);
    }
  }, [t]);

  useEffect(() => {
    loadPackages();
  }, [loadPackages, selectedAgent]);

  const createPackage = useCallback(
    async (packageKey: string, pkg: {
      package_id: string;
      path: string;
      description?: string;
      enabled?: boolean;
      bindings?: Record<string, string>;
    }) => {
      try {
        await api.createEmbodiment({ package_key: packageKey, package: pkg });
        message.success(t("embodiment.createSuccess"));
        await loadPackages();
        return true;
      } catch (error: any) {
        const errorMsg = error?.message || t("embodiment.createError");
        message.error(errorMsg);
        return false;
      }
    },
    [t, loadPackages],
  );

  const updatePackage = useCallback(
    async (packageKey: string, updates: {
      package_id?: string;
      path?: string;
      description?: string;
      enabled?: boolean;
      bindings?: Record<string, string>;
    }) => {
      try {
        await api.updateEmbodiment(packageKey, updates);
        message.success(t("embodiment.updateSuccess"));
        await loadPackages();
        return true;
      } catch (error: any) {
        const errorMsg = error?.message || t("embodiment.updateError");
        message.error(errorMsg);
        return false;
      }
    },
    [t, loadPackages],
  );

  const toggleEnabled = useCallback(
    async (pkg: EmbodimentPackageInfo) => {
      try {
        await api.toggleEmbodiment(pkg.key);
        message.success(
          pkg.enabled ? t("embodiment.disableSuccess") : t("embodiment.enableSuccess"),
        );
        await loadPackages();
      } catch (error) {
        message.error(t("embodiment.toggleError"));
      }
    },
    [t, loadPackages],
  );

  const deletePackage = useCallback(
    async (pkg: EmbodimentPackageInfo) => {
      try {
        await api.deleteEmbodiment(pkg.key);
        message.success(t("embodiment.deleteSuccess"));
        await loadPackages();
      } catch (error) {
        message.error(t("embodiment.deleteError"));
      }
    },
    [t, loadPackages],
  );

  return {
    packages,
    loading,
    createPackage,
    updatePackage,
    toggleEnabled,
    deletePackage,
  };
}
