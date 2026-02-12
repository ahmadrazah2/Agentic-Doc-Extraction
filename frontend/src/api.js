/**
 * API client for communicating with FastAPI backend
 */

import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

/**
 * Send a chat message
 * @param {string} message - User message
 * @returns {Promise} Response with answer and sources
 */
export const sendMessage = async (message) => {
    const response = await api.post('/chat', { message });
    return response.data;
};

/**
 * Upload a document
 * @param {File} file - File to upload
 * @returns {Promise} Response with upload status
 */
export const uploadDocument = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/upload', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });
    return response.data;
};

/**
 * Get chat history
 * @returns {Promise} Chat history
 */
export const getChatHistory = async () => {
    const response = await api.get('/history');
    return response.data;
};

/**
 * Get collection statistics
 * @returns {Promise} Collection stats
 */
export const getStats = async () => {
    const response = await api.get('/stats');
    return response.data;
};

export default api;
