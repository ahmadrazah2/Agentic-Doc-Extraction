import React, { useState, useRef, useEffect } from 'react';
import Message from './Message';
import { sendMessage, uploadDocument } from '../api';
import './ChatBox.css';

const ChatBox = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [uploading, setUploading] = useState(false);
    const messagesEndRef = useRef(null);
    const fileInputRef = useRef(null);

    // Auto-scroll to bottom
    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim() || loading) return;

        const userMessage = input.trim();
        setInput('');

        // Add user message
        setMessages(prev => [...prev, {
            role: 'user',
            content: userMessage,
            timestamp: new Date().toISOString()
        }]);

        setLoading(true);

        try {
            // Send to backend
            const response = await sendMessage(userMessage);

            // Add bot response
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: response.answer,
                timestamp: response.timestamp
            }]);
        } catch (error) {
            console.error('Error sending message:', error);
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `Error: ${error.response?.data?.detail || error.message}`,
                timestamp: new Date().toISOString()
            }]);
        } finally {
            setLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleFileUpload = async (e) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setUploading(true);

        // Add system message
        setMessages(prev => [...prev, {
            role: 'assistant',
            content: `📤 Uploading ${file.name}...`,
            timestamp: new Date().toISOString()
        }]);

        try {
            const response = await uploadDocument(file);

            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `✅ ${response.message}`,
                timestamp: new Date().toISOString()
            }]);
        } catch (error) {
            console.error('Error uploading file:', error);
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `❌ Upload failed: ${error.response?.data?.detail || error.message}`,
                timestamp: new Date().toISOString()
            }]);
        } finally {
            setUploading(false);
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
        }
    };

    return (
        <div className="chatbox-container">
            <div className="chatbox-header">
                <h1>🤖 RAG Chat Assistant</h1>
                <p>Ask questions about your documents</p>
            </div>

            <div className="messages-container">
                {messages.length === 0 && (
                    <div className="welcome-message">
                        <h2>👋 Welcome!</h2>
                        <p>Upload a document or start asking questions</p>
                    </div>
                )}

                {messages.map((msg, idx) => (
                    <Message
                        key={idx}
                        role={msg.role}
                        content={msg.content}
                        timestamp={msg.timestamp}
                    />
                ))}

                {loading && (
                    <div className="loading-indicator">
                        <div className="typing-dots">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            <div className="input-container">
                <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileUpload}
                    accept=".pdf,.png,.jpg,.jpeg"
                    style={{ display: 'none' }}
                />

                <button
                    className="upload-button"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploading}
                    title="Upload document"
                >
                    📎
                </button>

                <textarea
                    className="message-input"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Type your message..."
                    disabled={loading}
                    rows={1}
                />

                <button
                    className="send-button"
                    onClick={handleSend}
                    disabled={loading || !input.trim()}
                >
                    {loading ? '⏳' : '📤'}
                </button>
            </div>
        </div>
    );
};

export default ChatBox;
