"use client";

import React, { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import {
  streamChat,
  getIndexStatus,
  getDocuments,
  changeModel,
  SourceInfo,
} from "@/lib/api";
import ChatMessage from "@/components/ChatMessage";
import Link from "next/link";

export default function ChatPage() {
  const router = useRouter();
  const { user, isAuthenticated, loading, signOut, isAdmin } = useAuth();

  const [messages, setMessages] = useState<
    { role: string; content: string; sources?: SourceInfo[] }[]
  >([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentStream, setCurrentStream] = useState("");
  const [currentSources, setCurrentSources] = useState<SourceInfo[]>([]);

  const [indexReady, setIndexReady] = useState(false);
  const [docCount, setDocCount] = useState(0);
  const [currentModel, setCurrentModel] = useState("llama3.2:3b");

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const streamRef = useRef("");
  const sourcesRef = useRef<SourceInfo[]>([]);

  useEffect(() => {
    if (!loading && !isAuthenticated) {
      router.replace("/auth");
    }
  }, [loading, isAuthenticated, router]);

  useEffect(() => {
    if (isAuthenticated) {
      fetchStatus();
    }
  }, [isAuthenticated]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentStream]);

  const fetchStatus = async () => {
    try {
      const status = await getIndexStatus();
      setIndexReady(status.indexed);
      setDocCount(status.document_count);
      if (status.current_model) setCurrentModel(status.current_model);
    } catch (e) {
      console.error("Failed to fetch index status", e);
    }
  };

  const handleModelChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newModel = e.target.value;
    try {
      await changeModel(newModel);
      setCurrentModel(newModel);
      alert(`Model changed to ${newModel}`);
    } catch (err: any) {
      alert(`Failed to change model: ${err.message}`);
    }
  };

  const handleSubmitRobust = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming || !indexReady) return;

    const userMsg = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMsg }]);

    setIsStreaming(true);
    setCurrentStream("");
    streamRef.current = "";
    sourcesRef.current = [];

    try {
      await streamChat(
        userMsg,
        messages.map((m) => ({ role: m.role, content: m.content })),
        (token) => {
          streamRef.current += token;
          setCurrentStream(streamRef.current);
        },
        (sources) => {
          sourcesRef.current = sources;
        },
        () => {
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: streamRef.current,
              sources: sourcesRef.current,
            },
          ]);
          setIsStreaming(false);
          setCurrentStream("");
        },
        (err) => {
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: `*Error: ${err}*` },
          ]);
          setIsStreaming(false);
          setCurrentStream("");
        },
      );
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `*Connection Error: ${err.message}*` },
      ]);
      setIsStreaming(false);
    }
  };

  if (loading || !user) {
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
        display: "flex",
        height: "100vh",
        width: "100%",
        overflow: "hidden",
        background: "var(--bg-deep)",
      }}
    >
      {/* ── Sidebar ── */}
      <div
        style={{
          width: "var(--sidebar-width)",
          minWidth: "var(--sidebar-width)",
          background: "var(--bg-panel)",
          borderRight: "1px solid var(--border-glass)",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
          zIndex: 10,
        }}
      >
        {/* Sidebar Header */}
        <div
          style={{
            padding: "1.25rem 1.5rem",
            borderBottom: "1px solid var(--border-glass)",
            flexShrink: 0,
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.5rem",
              marginBottom: "1rem",
            }}
          >
            <span
              style={{
                fontWeight: 700,
                fontSize: "1.1rem",
                color: "var(--text-primary)",
              }}
            >
              Retail KB
            </span>
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              marginBottom: "0.75rem",
            }}
          >
            <div
              style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}
            >
              <span
                style={{
                  fontSize: "0.9rem",
                  color: "var(--text-secondary)",
                  fontWeight: 500,
                }}
              >
                {user.username}
              </span>
              <span
                className={`badge ${isAdmin ? "badge-admin" : "badge-user"}`}
              >
                {user.role}
              </span>
            </div>
          </div>

          <button
            onClick={signOut}
            className="glass-button w-full"
            style={{ padding: "0.5rem", fontSize: "0.875rem" }}
          >
            Sign Out
          </button>
        </div>

        {/* Sidebar Scrollable Body */}
        <div style={{ flex: 1, overflowY: "auto", padding: "1.25rem 1.5rem" }}>
          <div style={{ marginBottom: "1.5rem" }}>
            <label
              style={{
                fontSize: "0.8rem",
                fontWeight: 600,
                color: "var(--text-muted)",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                display: "block",
                marginBottom: "0.5rem",
              }}
            >
              LLM Model
            </label>
            <select
              className="glass-input"
              value={currentModel}
              onChange={handleModelChange}
              style={{ padding: "0.5rem 0.75rem", fontSize: "0.875rem" }}
            >
              <option value="llama3.2:3b">llama3.2:3b</option>
              <option value="qwen2.5:7b">qwen2.5:7b</option>
              <option value="llama3.1:8b">llama3.1:8b</option>
              <option value="mistral:7b">mistral:7b</option>
              <option value="gemma2:2b">gemma2:2b</option>
            </select>
          </div>

          <div
            style={{
              marginBottom: "1.5rem",
              background: "var(--bg-dark)",
              padding: "1rem",
              borderRadius: "10px",
            }}
          >
            <div
              style={{
                fontSize: "0.8rem",
                fontWeight: 600,
                color: "var(--text-muted)",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                marginBottom: "0.75rem",
              }}
            >
              Index Status
            </div>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "0.5rem",
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
                  fontWeight: 500,
                  display: "flex",
                  alignItems: "center",
                }}
              >
                <span
                  className={`status-dot ${indexReady ? "ready" : "not-ready"}`}
                ></span>
                {indexReady ? "Ready" : "Not built"}
              </span>
            </div>
          </div>

          <button
            onClick={() => setMessages([])}
            className="glass-button w-full"
            style={{ marginBottom: "1rem", fontSize: "0.875rem" }}
          >
            Clear Chat
          </button>

          {isAdmin && (
            <div
              style={{
                paddingTop: "1rem",
                borderTop: "1px solid var(--border-glass)",
              }}
            >
              <div
                style={{
                  fontSize: "0.8rem",
                  fontWeight: 600,
                  color: "var(--accent-orange)",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  marginBottom: "0.75rem",
                }}
              >
                Admin
              </div>
              <Link
                href="/admin"
                style={{ display: "block", textDecoration: "none" }}
              >
                <button
                  className="primary-button w-full"
                  style={{ fontSize: "0.875rem" }}
                >
                  Admin Dashboard →
                </button>
              </Link>
            </div>
          )}
        </div>
      </div>

      {/* ── Main Chat Area ── */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
          minWidth: 0,
        }}
      >
        {/* Header – fixed within flow, not absolutely positioned */}
        <div
          style={{
            padding: "1.25rem 2rem",
            background: "var(--bg-panel)",
            borderBottom: "1px solid var(--border-glass)",
            flexShrink: 0,
            zIndex: 5,
          }}
        >
          <h1
            style={{
              fontSize: "1.2rem",
              margin: 0,
              fontWeight: 600,
              color: "var(--text-primary)",
            }}
          >
            Retail Knowledge Assistant
          </h1>
          <div
            style={{
              fontSize: "0.8rem",
              color: "var(--text-muted)",
              marginTop: "0.2rem",
            }}
          >
            Ask questions about retail operations, scenarios, terms, and
            processes
          </div>
        </div>

        {/* Messages – scrollable area that fills the remaining space */}
        <div style={{ flex: 1, overflowY: "auto", padding: "1.5rem 2rem" }}>
          {!indexReady && (
            <div
              className="glass-panel"
              style={{
                padding: "1rem",
                borderLeft: "4px solid var(--accent-orange)",
                marginBottom: "1.5rem",
              }}
            >
              <span style={{ fontSize: "0.9rem" }}>
                ⚠️ The knowledge base is not ready yet.{" "}
                {isAdmin
                  ? "Go to the Admin Dashboard to build the index."
                  : "Please ask an admin to build the index."}
              </span>
            </div>
          )}

          {messages.length === 0 && !isStreaming ? (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                height: "100%",
                opacity: 0.4,
                textAlign: "center",
                userSelect: "none",
              }}
            >
              <div style={{ fontSize: "2.5rem", marginBottom: "1rem" }}>💬</div>
              <h2 style={{ fontSize: "1.25rem", marginBottom: "0.5rem" }}>
                How can I help you today?
              </h2>
              <p style={{ fontSize: "0.9rem", color: "var(--text-muted)" }}>
                Try asking about SKUs, planograms, or stockouts.
              </p>
            </div>
          ) : (
            messages.map((m, i) => (
              <ChatMessage
                key={i}
                role={m.role}
                content={m.content}
                sources={m.sources}
              />
            ))
          )}

          {isStreaming && (
            <ChatMessage
              role="assistant"
              content={currentStream}
              isStreaming={true}
            />
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar – flush at bottom, in flow */}
        <div
          style={{
            padding: "1rem 2rem 1.25rem",
            background: "var(--bg-panel)",
            borderTop: "1px solid var(--border-glass)",
            flexShrink: 0,
          }}
        >
          <form
            onSubmit={handleSubmitRobust}
            style={{
              position: "relative",
              maxWidth: "800px",
              margin: "0 auto",
            }}
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={
                indexReady
                  ? "Ask about retail..."
                  : "Index not ready — contact an admin"
              }
              disabled={isStreaming || !indexReady}
              className="glass-input"
              style={{
                padding: "0.9rem 5rem 0.9rem 1.25rem",
                fontSize: "0.95rem",
                borderRadius: "24px",
              }}
            />
            <button
              type="submit"
              disabled={isStreaming || !indexReady || !input.trim()}
              className="primary-button"
              style={{
                position: "absolute",
                right: "6px",
                top: "6px",
                bottom: "6px",
                padding: "0 1.25rem",
                borderRadius: "18px",
                fontSize: "1.1rem",
              }}
            >
              ↑
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
