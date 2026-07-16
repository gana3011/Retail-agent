'use client';

import React, { useState } from 'react';
import { SourceInfo } from '@/lib/api';

interface ChatMessageProps {
  role: string;
  content: string;
  sources?: SourceInfo[];
  isStreaming?: boolean;
}

export default function ChatMessage({ role, content, sources, isStreaming }: ChatMessageProps) {
  const isUser = role === 'user';
  const [showSources, setShowSources] = useState(false);

  // Very basic markdown rendering for bold and code
  const renderContent = (text: string) => {
    // Escape HTML first
    let safeText = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    
    // Bold: **text**
    safeText = safeText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Inline code: `text`
    safeText = safeText.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Line breaks
    safeText = safeText.replace(/\n/g, '<br/>');
    
    return <div className="prose" dangerouslySetInnerHTML={{ __html: safeText }} />;
  };

  return (
    <div className={`flex w-full mb-6 ${isUser ? 'justify-end' : 'justify-start'} animate-slide-up`}>
      <div 
        style={{ 
          maxWidth: '80%', 
          padding: '1.25rem',
          borderRadius: '20px',
          borderBottomRightRadius: isUser ? '4px' : '20px',
          borderBottomLeftRadius: !isUser ? '4px' : '20px',
          background: isUser ? 'var(--gradient-primary)' : 'var(--bg-panel)',
          border: isUser ? 'none' : '1px solid var(--border-glass)',
          boxShadow: isUser ? '0 4px 6px -1px rgba(0, 0, 0, 0.1)' : '0 1px 3px rgba(0,0,0,0.05)',
          color: isUser ? 'white' : 'var(--text-primary)'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.5rem', opacity: 0.8, fontSize: '0.85rem', gap: '0.5rem' }}>
          {isUser ? 'You' : 'Retail Assistant'}
          {isStreaming && !isUser && <span style={{ fontStyle: 'italic', color: 'var(--text-muted)' }}>typing...</span>}
        </div>
        
        {renderContent(content)}
        
        {isStreaming && content === '' && (
          <div className="typing-indicator mt-2">
            <div className="typing-dot"></div>
            <div className="typing-dot"></div>
            <div className="typing-dot"></div>
          </div>
        )}

        {sources && sources.length > 0 && !isStreaming && (
          <div className="mt-4 border-t" style={{ borderColor: 'var(--border-glass)', paddingTop: '1rem' }}>
            <button 
              onClick={() => setShowSources(!showSources)}
              style={{ background: 'none', border: 'none', color: 'var(--accent-blue)', cursor: 'pointer', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: 0 }}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ transform: showSources ? 'rotate(180deg)' : 'rotate(0)', transition: 'transform 0.2s' }}>
                <polyline points="6 9 12 15 18 9"></polyline>
              </svg>
              {showSources ? 'Hide Sources' : `View Sources (${sources.length})`}
            </button>
            
            {showSources && (
              <div className="mt-3 flex flex-col gap-2 animate-fade-in">
                {sources.map((s, i) => (
                  <div key={i} style={{ background: 'var(--bg-dark)', padding: '0.75rem', borderRadius: '8px', fontSize: '0.85rem' }}>
                    <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>{s.source_doc}</div>
                    <div className="flex flex-col gap-1 text-muted">
                      {s.title && <div><span style={{color: 'var(--text-secondary)'}}>Scenario:</span> {s.title}</div>}
                      {s.term && <div><span style={{color: 'var(--text-secondary)'}}>Term:</span> {s.term}</div>}
                      {s.domain && <div><span style={{color: 'var(--text-secondary)'}}>Domain:</span> {s.domain}</div>}
                      <div><span style={{color: 'var(--text-secondary)'}}>Score:</span> {s.relevance_score.toFixed(3)}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
