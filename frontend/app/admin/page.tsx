"use client";

import React, { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/lib/auth-context";
import {
  getUsers,
  changeUserRole,
  deleteUser,
  getDocuments,
  uploadDocuments,
  deleteDocuments,
  getIndexStatus,
  rebuildIndex,
  fullRebuildIndex,
  fixLock,
} from "@/lib/api";

export default function AdminPage() {
  const router = useRouter();
  const { user, isAuthenticated, loading, isAdmin } = useAuth();

  const [users, setUsers] = useState<{ username: string; role: string }[]>([]);
  const [docs, setDocs] = useState<{ filename: string; size_bytes: number }[]>(
    [],
  );

  const [indexStatus, setIndexStatus] = useState({
    indexed: false,
    docCount: 0,
    vecCount: 0,
    model: "",
  });

  const [isProcessing, setIsProcessing] = useState(false);
  const [toast, setToast] = useState<{
    message: string;
    type: "success" | "error";
  } | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedDocs, setSelectedDocs] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (!loading) {
      if (!isAuthenticated) router.replace("/auth");
      else if (!isAdmin) router.replace("/chat");
      else {
        fetchData();
      }
    }
  }, [loading, isAuthenticated, isAdmin, router]);

  const showToast = (
    message: string,
    type: "success" | "error" = "success",
  ) => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3000);
  };

  const fetchData = async () => {
    try {
      const [u, d, s] = await Promise.all([
        getUsers(),
        getDocuments(),
        getIndexStatus(),
      ]);
      setUsers(u);
      setDocs(d.documents);
      setIndexStatus({
        indexed: s.indexed,
        docCount: s.document_count,
        vecCount: s.vector_count,
        model: s.current_model,
      });
    } catch (e: any) {
      showToast(e.message || "Failed to fetch admin data", "error");
    }
  };

  // ── User Handlers ──
  const handleRoleChange = async (username: string, newRole: string) => {
    try {
      setIsProcessing(true);
      await changeUserRole(username, newRole);
      showToast(`Role updated for ${username}`);
      await fetchData();
    } catch (e: any) {
      showToast(e.message, "error");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDeleteUser = async (username: string) => {
    if (!confirm(`Are you sure you want to delete ${username}?`)) return;
    try {
      setIsProcessing(true);
      await deleteUser(username);
      showToast(`User ${username} deleted`);
      await fetchData();
    } catch (e: any) {
      showToast(e.message, "error");
    } finally {
      setIsProcessing(false);
    }
  };

  // ── Document Handlers ──
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;

    try {
      setIsProcessing(true);
      showToast(
        "Uploading and rebuilding index... this may take a minute.",
        "success",
      );
      const res = await uploadDocuments(e.target.files);
      showToast(res.message);
      await fetchData();
    } catch (err: any) {
      showToast(err.message, "error");
    } finally {
      setIsProcessing(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const toggleDocSelection = (filename: string) => {
    const next = new Set(selectedDocs);
    if (next.has(filename)) next.delete(filename);
    else next.add(filename);
    setSelectedDocs(next);
  };

  const handleDeleteDocs = async () => {
    if (selectedDocs.size === 0) return;
    if (!confirm(`Delete ${selectedDocs.size} document(s) and rebuild index?`))
      return;

    try {
      setIsProcessing(true);
      showToast("Deleting and rebuilding index...", "success");
      const res = await deleteDocuments(Array.from(selectedDocs));
      showToast(res.message);
      setSelectedDocs(new Set());
      await fetchData();
    } catch (err: any) {
      showToast(err.message, "error");
    } finally {
      setIsProcessing(false);
    }
  };

  // ── Index Handlers ──
  const handleRebuild = async (full: boolean) => {
    try {
      setIsProcessing(true);
      showToast(`${full ? "Full " : ""}Rebuild started...`, "success");
      const res = full ? await fullRebuildIndex() : await rebuildIndex();
      showToast(res.message);
      await fetchData();
    } catch (err: any) {
      showToast(err.message, "error");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFixLock = async () => {
    try {
      setIsProcessing(true);
      const res = await fixLock();
      showToast(res.message);
      await fetchData();
    } catch (err: any) {
      showToast(err.message, "error");
    } finally {
      setIsProcessing(false);
    }
  };

  if (loading || !user || !isAdmin) {
    return (
      <div
        style={{
          height: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <div className="typing-indicator">
          <div className="typing-dot"></div>
          <div className="typing-dot"></div>
          <div className="typing-dot"></div>
        </div>
      </div>
    );
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        padding: "2rem 1.5rem",
        background: "var(--bg-deep)",
        overflowY: "auto",
      }}
    >
      <div style={{ maxWidth: "1100px", margin: "0 auto" }}>
        {/* Page Header */}
        <div
          style={{
            display: "flex",
            alignItems: "flex-start",
            justifyContent: "space-between",
            marginBottom: "2rem",
          }}
          className="animate-slide-up"
        >
          <div>
            <Link
              href="/chat"
              style={{
                color: "var(--text-muted)",
                marginBottom: "0.5rem",
                display: "inline-flex",
                alignItems: "center",
                gap: "0.25rem",
                fontSize: "0.875rem",
              }}
            >
              ← Back to Chat
            </Link>
            <h1
              style={{
                color: "var(--text-primary)",
                fontSize: "2rem",
                fontWeight: 700,
                margin: 0,
              }}
            >
              Admin Dashboard
            </h1>
          </div>
          <span
            className="badge badge-admin"
            style={{ padding: "0.4rem 0.9rem", marginTop: "0.5rem" }}
          >
            Admin Session
          </span>
        </div>

        {/* Top Row: Index + Documents side by side */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(280px, 360px) 1fr",
            gap: "1.5rem",
            marginBottom: "1.5rem",
            alignItems: "start",
          }}
        >
          {/* Index Management */}
          <div
            className="glass-panel animate-slide-up"
            style={{ padding: "1.75rem", animationDelay: "0.1s" }}
          >
            <h2
              style={{
                fontSize: "1.1rem",
                fontWeight: 600,
                marginBottom: "1rem",
                color: "var(--text-primary)",
              }}
            >
              Index Management
            </h2>

            <div
              style={{
                background: "var(--bg-dark)",
                padding: "1rem",
                borderRadius: "10px",
                marginBottom: "1.25rem",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: "0.6rem",
                }}
              >
                <span
                  style={{ fontSize: "0.875rem", color: "var(--text-muted)" }}
                >
                  Status
                </span>
                <span
                  style={{
                    fontSize: "0.875rem",
                    fontWeight: 600,
                    display: "flex",
                    alignItems: "center",
                  }}
                >
                  <span
                    className={`status-dot ${indexStatus.indexed ? "ready" : "not-ready"}`}
                  ></span>
                  {indexStatus.indexed ? "Ready" : "Not built"}
                </span>
              </div>

              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: "0.6rem",
                }}
              >
                <span
                  style={{ fontSize: "0.875rem", color: "var(--text-muted)" }}
                >
                  Vectors
                </span>
                <span style={{ fontSize: "0.875rem", fontWeight: 600 }}>
                  {indexStatus.vecCount}
                </span>
              </div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <span
                  style={{ fontSize: "0.875rem", color: "var(--text-muted)" }}
                >
                  Active Model
                </span>
                <span style={{ fontSize: "0.875rem", fontWeight: 600 }}>
                  {indexStatus.model || "—"}
                </span>
              </div>
            </div>

            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "0.75rem",
              }}
            >
              <button
                onClick={() => handleRebuild(false)}
                disabled={isProcessing}
                className="glass-button"
                style={{ fontSize: "0.875rem" }}
              >
                Rebuild Index (Phase 1+2)
              </button>
              <button
                onClick={() => handleRebuild(true)}
                disabled={isProcessing}
                className="primary-button"
                style={{ fontSize: "0.875rem" }}
              >
                Full Rebuild (Phase 0+1+2)
              </button>
              <button
                onClick={handleFixLock}
                disabled={isProcessing}
                className="glass-button"
                style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}
              >
                Fix Qdrant Lock
              </button>
            </div>
          </div>

          {/* Document Management */}
          <div
            className="glass-panel animate-slide-up"
            style={{ padding: "1.75rem", animationDelay: "0.2s" }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: "1rem",
              }}
            >
              <h2
                style={{
                  fontSize: "1.1rem",
                  fontWeight: 600,
                  margin: 0,
                  color: "var(--text-primary)",
                }}
              >
                Document Management
              </h2>
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                {docs.length} total
              </span>
            </div>

            <div
              style={{
                border: "2px dashed var(--border-glass)",
                borderRadius: "10px",
                padding: "1.5rem",
                textAlign: "center",
                background: "var(--bg-dark)",
                cursor: "pointer",
                marginBottom: "1.25rem",
                transition: "border-color 0.2s",
              }}
              onClick={() => fileInputRef.current?.click()}
              onMouseEnter={(e) =>
                (e.currentTarget.style.borderColor = "#94a3b8")
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.borderColor = "var(--border-glass)")
              }
            >
              <div style={{ fontSize: "1.5rem", marginBottom: "0.5rem" }}>
                📁
              </div>
              <div
                style={{
                  fontWeight: 600,
                  fontSize: "0.9rem",
                  marginBottom: "0.25rem",
                  color: "var(--text-primary)",
                }}
              >
                Click to upload .docx files
              </div>
              <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                Files will be processed and indexed automatically
              </div>
              <input
                type="file"
                ref={fileInputRef}
                accept=".docx"
                multiple
                style={{ display: "none" }}
                onChange={handleUpload}
              />
            </div>

            {docs.length > 0 && (
              <div>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: "0.75rem",
                    paddingBottom: "0.75rem",
                    borderBottom: "1px solid var(--border-glass)",
                  }}
                >
                  <span
                    style={{
                      fontWeight: 600,
                      fontSize: "0.9rem",
                      color: "var(--text-primary)",
                    }}
                  >
                    Existing Documents
                  </span>
                  {selectedDocs.size > 0 && (
                    <button
                      onClick={handleDeleteDocs}
                      disabled={isProcessing}
                      className="glass-button"
                      style={{
                        padding: "0.3rem 0.75rem",
                        fontSize: "0.8rem",
                        color: "var(--accent-red)",
                        borderColor: "var(--accent-red)",
                      }}
                    >
                      Delete Selected ({selectedDocs.size})
                    </button>
                  )}
                </div>
                <div
                  style={{
                    maxHeight: "220px",
                    overflowY: "auto",
                    display: "flex",
                    flexDirection: "column",
                    gap: "0.5rem",
                    paddingRight: "0.25rem",
                  }}
                >
                  {docs.map((d) => (
                    <div
                      key={d.filename}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        padding: "0.6rem 0.75rem",
                        borderRadius: "8px",
                        background: "var(--bg-dark)",
                      }}
                    >
                      <label
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "0.75rem",
                          cursor: "pointer",
                          flex: 1,
                          minWidth: 0,
                        }}
                      >
                        <input
                          type="checkbox"
                          checked={selectedDocs.has(d.filename)}
                          onChange={() => toggleDocSelection(d.filename)}
                          style={{
                            accentColor: "var(--accent-red)",
                            flexShrink: 0,
                          }}
                        />
                        <span
                          style={{
                            fontSize: "0.85rem",
                            whiteSpace: "nowrap",
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            color: "var(--text-secondary)",
                          }}
                        >
                          {d.filename}
                        </span>
                      </label>
                      <span
                        style={{
                          fontSize: "0.75rem",
                          color: "var(--text-muted)",
                          flexShrink: 0,
                          marginLeft: "0.5rem",
                        }}
                      >
                        {(d.size_bytes / 1024).toFixed(1)} KB
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Full-width: User Management */}
        <div
          className="glass-panel animate-slide-up"
          style={{ padding: "1.75rem", animationDelay: "0.3s" }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              marginBottom: "1.25rem",
            }}
          >
            <h2
              style={{
                fontSize: "1.1rem",
                fontWeight: 600,
                margin: 0,
                color: "var(--text-primary)",
              }}
            >
              👥 User Management
            </h2>
            <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
              {users.length} users
            </span>
          </div>

          <div style={{ overflowX: "auto" }}>
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                textAlign: "left",
              }}
            >
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border-glass)" }}>
                  <th
                    style={{
                      padding: "0.75rem 1rem",
                      color: "var(--text-muted)",
                      fontSize: "0.8rem",
                      fontWeight: 600,
                      textTransform: "uppercase",
                      letterSpacing: "0.05em",
                    }}
                  >
                    Username
                  </th>
                  <th
                    style={{
                      padding: "0.75rem 1rem",
                      color: "var(--text-muted)",
                      fontSize: "0.8rem",
                      fontWeight: 600,
                      textTransform: "uppercase",
                      letterSpacing: "0.05em",
                    }}
                  >
                    Role
                  </th>
                  <th
                    style={{
                      padding: "0.75rem 1rem",
                      color: "var(--text-muted)",
                      fontSize: "0.8rem",
                      fontWeight: 600,
                      textTransform: "uppercase",
                      letterSpacing: "0.05em",
                      textAlign: "right",
                    }}
                  >
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {users.map((u) => (
                  <tr
                    key={u.username}
                    style={{ borderBottom: "1px solid var(--border-glass)" }}
                  >
                    <td
                      style={{
                        padding: "0.875rem 1rem",
                        fontWeight: 500,
                        color: "var(--text-primary)",
                        fontSize: "0.9rem",
                      }}
                    >
                      {u.username}
                      {u.username === user.username && (
                        <span
                          style={{
                            fontSize: "0.75rem",
                            color: "var(--text-muted)",
                            marginLeft: "0.5rem",
                          }}
                        >
                          (You)
                        </span>
                      )}
                    </td>
                    <td style={{ padding: "0.875rem 1rem" }}>
                      <select
                        value={u.role}
                        onChange={(e) =>
                          handleRoleChange(u.username, e.target.value)
                        }
                        disabled={isProcessing || u.username === user.username}
                        className="glass-input"
                        style={{
                          padding: "0.4rem 0.75rem",
                          width: "auto",
                          fontSize: "0.875rem",
                        }}
                      >
                        <option value="user">User</option>
                        <option value="admin">Admin</option>
                      </select>
                    </td>
                    <td
                      style={{ padding: "0.875rem 1rem", textAlign: "right" }}
                    >
                      <button
                        onClick={() => handleDeleteUser(u.username)}
                        disabled={isProcessing || u.username === user.username}
                        className="glass-button"
                        style={{
                          padding: "0.4rem 0.875rem",
                          fontSize: "0.875rem",
                          color:
                            u.username === user.username
                              ? "var(--text-muted)"
                              : "var(--accent-red)",
                          borderColor:
                            u.username === user.username
                              ? "var(--border-glass)"
                              : "var(--accent-red)",
                        }}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Toast Notification */}
      {toast && (
        <div className={`toast ${toast.type}`}>
          <span>{toast.type === "success" ? "✓" : "✕"}</span>
          {toast.message}
        </div>
      )}
    </div>
  );
}
