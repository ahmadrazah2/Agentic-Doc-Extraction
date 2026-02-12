import React from 'react';
import './Message.css';

const Message = ({ role, content, timestamp }) => {
    const isUser = role === 'user';

    return (
        <div className={`message ${isUser ? 'user-message' : 'bot-message'}`}>
            <div className="message-header">
                <span className="message-role">{isUser ? '👤 You' : '🤖 Assistant'}</span>
                {timestamp && (
                    <span className="message-time">
                        {new Date(timestamp).toLocaleTimeString()}
                    </span>
                )}
            </div>
            <div className="message-content">
                {content}
            </div>
        </div>
    );
};

export default Message;
