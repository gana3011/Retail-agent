"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";

export default function AuthPage() {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const router = useRouter();
  const { signIn, signUp } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    setIsLoading(true);

    try {
      if (isLogin) {
        await signIn(username, password);
        router.push("/chat");
      } else {
        if (password !== confirmPassword) {
          throw new Error("Passwords do not match");
        }
        const msg = await signUp(username, password);
        setSuccess(msg + " You can now sign in.");
        setIsLogin(true);
        setPassword("");
        setConfirmPassword("");
      }
    } catch (err: any) {
      setError(err.message || "An error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "1.5rem",
        background: "var(--bg-deep)",
      }}
    >
      {/* Branding Header */}
      <div
        className="text-center mb-8 animate-slide-up"
        style={{ animationDelay: "0.1s" }}
      >
        <h1
          style={{
            fontSize: "2.25rem",
            fontWeight: 700,
            color: "var(--text-primary)",
            marginBottom: "0.5rem",
          }}
        >
          Retail Knowledge Bot
        </h1>
        <p style={{ color: "var(--text-muted)", margin: 0 }}>
          Your AI-powered retail intelligence assistant
        </p>
      </div>

      {/* Auth Card */}
      <div
        className="glass-panel animate-slide-up"
        style={{
          width: "100%",
          maxWidth: "440px",
          padding: "2rem",
          animationDelay: "0.2s",
        }}
      >
        {/* Tabs */}
        <div
          style={{
            display: "flex",
            gap: "0",
            marginBottom: "1.75rem",
            background: "var(--bg-dark)",
            borderRadius: "8px",
            padding: "4px",
          }}
        >
          <button
            type="button"
            onClick={() => {
              setIsLogin(true);
              setError("");
              setSuccess("");
            }}
            style={{
              flex: 1,
              background: isLogin ? "var(--bg-panel)" : "transparent",
              border: "none",
              padding: "0.6rem 1rem",
              cursor: "pointer",
              color: isLogin ? "var(--text-primary)" : "var(--text-muted)",
              fontWeight: isLogin ? 600 : 400,
              borderRadius: "6px",
              transition: "all 0.2s",
              fontSize: "0.95rem",
              fontFamily: "inherit",
              boxShadow: isLogin ? "0 1px 3px rgba(0,0,0,0.08)" : "none",
            }}
          >
            Sign In
          </button>
          <button
            type="button"
            onClick={() => {
              setIsLogin(false);
              setError("");
              setSuccess("");
            }}
            style={{
              flex: 1,
              background: !isLogin ? "var(--bg-panel)" : "transparent",
              border: "none",
              padding: "0.6rem 1rem",
              cursor: "pointer",
              color: !isLogin ? "var(--text-primary)" : "var(--text-muted)",
              fontWeight: !isLogin ? 600 : 400,
              borderRadius: "6px",
              transition: "all 0.2s",
              fontSize: "0.95rem",
              fontFamily: "inherit",
              boxShadow: !isLogin ? "0 1px 3px rgba(0,0,0,0.08)" : "none",
            }}
          >
            Sign Up
          </button>
        </div>

        {/* Form */}
        <form
          onSubmit={handleSubmit}
          style={{ display: "flex", flexDirection: "column", gap: "1rem" }}
        >
          <div>
            <label
              style={{
                display: "block",
                marginBottom: "0.4rem",
                fontSize: "0.875rem",
                fontWeight: 500,
                color: "var(--text-secondary)",
              }}
            >
              Username
            </label>
            <input
              type="text"
              className="glass-input"
              placeholder="Enter username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>

          <div>
            <label
              style={{
                display: "block",
                marginBottom: "0.4rem",
                fontSize: "0.875rem",
                fontWeight: 500,
                color: "var(--text-secondary)",
              }}
            >
              Password
            </label>
            <input
              type="password"
              className="glass-input"
              placeholder={isLogin ? "Enter password" : "Min. 6 characters"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          {!isLogin && (
            <div className="animate-fade-in">
              <label
                style={{
                  display: "block",
                  marginBottom: "0.4rem",
                  fontSize: "0.875rem",
                  fontWeight: 500,
                  color: "var(--text-secondary)",
                }}
              >
                Confirm Password
              </label>
              <input
                type="password"
                className="glass-input"
                placeholder="Repeat password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
              />
            </div>
          )}

          <button
            type="submit"
            className="primary-button"
            disabled={isLoading}
            style={{
              padding: "0.9rem",
              fontSize: "1rem",
              marginTop: "0.5rem",
              borderRadius: "8px",
            }}
          >
            {isLoading
              ? "Processing..."
              : isLogin
                ? "Sign In →"
                : "Create Account →"}
          </button>
        </form>

        {/* Messages */}
        {error && (
          <div
            className="mt-4 animate-fade-in"
            style={{
              padding: "0.75rem 1rem",
              borderRadius: "6px",
              fontSize: "0.875rem",
              background: "rgba(239,68,68,0.08)",
              borderLeft: "3px solid var(--accent-red)",
              color: "#b91c1c",
            }}
          >
            {error}
          </div>
        )}
        {success && (
          <div
            className="mt-4 animate-fade-in"
            style={{
              padding: "0.75rem 1rem",
              borderRadius: "6px",
              fontSize: "0.875rem",
              background: "rgba(16,185,129,0.08)",
              borderLeft: "3px solid #10b981",
              color: "#065f46",
            }}
          >
            {success}
          </div>
        )}
      </div>

      <div
        className="mt-8 animate-fade-in"
        style={{
          animationDelay: "0.4s",
          color: "var(--text-muted)",
          fontSize: "0.8rem",
          textAlign: "center",
        }}
      >
        Default admin:{" "}
        <code
          style={{
            background: "var(--bg-dark)",
            padding: "2px 6px",
            borderRadius: "4px",
            color: "var(--text-primary)",
          }}
        >
          admin
        </code>{" "}
        /{" "}
        <code
          style={{
            background: "var(--bg-dark)",
            padding: "2px 6px",
            borderRadius: "4px",
            color: "var(--text-primary)",
          }}
        >
          Admin@1234
        </code>
      </div>
    </div>
  );
}
