// frontend/src/components/AudioRecorder.jsx
import React, { useState, useEffect, useRef } from 'react';
import AudioStreamService from '../services/websocket';

const AudioRecorder = () => {
    const [isConnected, setIsConnected] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [inferenceTime, setInferenceTime] = useState(0);
    const [audioProcessed, setAudioProcessed] = useState(0);
    const [error, setError] = useState(null);
    
    const audioServiceRef = useRef(null);
    const serverUrl = process.env.REACT_APP_SERVER_URL || 'ws://localhost:8000';
    
    useEffect(() => {
        audioServiceRef.current = new AudioStreamService();
        
        // Set up message handlers
        audioServiceRef.current.on('config', (message) => {
            console.log('Received config:', message);
        });
        
        audioServiceRef.current.on('audio', (message) => {
            // Play converted audio
            audioServiceRef.current.playAudio(message.data);
            setInferenceTime(message.inference_time_ms);
            setAudioProcessed(prev => prev + 1);
        });
        
        audioServiceRef.current.on('error', (message) => {
            setError(message.message);
        });
        
        return () => {
            if (audioServiceRef.current) {
                audioServiceRef.current.disconnect();
            }
        };
    }, []);
    
    const handleConnect = async () => {
        try {
            await audioServiceRef.current.connect(serverUrl);
            setIsConnected(true);
            setError(null);
        } catch (err) {
            setError('Failed to connect to server');
            console.error(err);
        }
    };
    
    const handleStartRecording = async () => {
        try {
            await audioServiceRef.current.startRecording();
            setIsRecording(true);
            setError(null);
        } catch (err) {
            setError('Failed to start recording');
            console.error(err);
        }
    };
    
    const handleStopRecording = () => {
        audioServiceRef.current.stopRecording();
        setIsRecording(false);
    };
    
    const handleDisconnect = () => {
        audioServiceRef.current.disconnect();
        setIsConnected(false);
        setIsRecording(false);
    };
    
    return (
        <div className="audio-recorder-container">
            <h2>Real-Time Dysarthric Speech Conversion</h2>
            
            <div className="status-panel">
                <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
                    {isConnected ? '● Connected' : '○ Disconnected'}
                </div>
                <div className={`status-indicator ${isRecording ? 'recording' : ''}`}>
                    {isRecording ? '● Recording' : '○ Not Recording'}
                </div>
            </div>
            
            <div className="controls">
                {!isConnected ? (
                    <button onClick={handleConnect} className="btn btn-primary">
                        Connect to Server
                    </button>
                ) : (
                    <>
                        {!isRecording ? (
                            <button onClick={handleStartRecording} className="btn btn-success">
                                Start Recording
                            </button>
                        ) : (
                            <button onClick={handleStopRecording} className="btn btn-danger">
                                Stop Recording
                            </button>
                        )}
                        <button onClick={handleDisconnect} className="btn btn-secondary">
                            Disconnect
                        </button>
                    </>
                )}
            </div>
            
            <div className="metrics">
                <div className="metric-card">
                    <h4>Inference Time</h4>
                    <p>{inferenceTime.toFixed(2)} ms</p>
                </div>
                <div className="metric-card">
                    <h4>Chunks Processed</h4>
                    <p>{audioProcessed}</p>
                </div>
                <div className="metric-card">
                    <h4>Real-Time Factor</h4>
                    <p>{inferenceTime > 0 ? ((inferenceTime / 256) * 16).toFixed(2) : '0.00'}</p>
                </div>
            </div>
            
            {error && (
                <div className="error-message">
                    <strong>Error:</strong> {error}
                </div>
            )}
            
            <div className="info-panel">
                <h3>How to Use:</h3>
                <ol>
                    <li>Click "Connect to Server" to establish connection</li>
                    <li>Click "Start Recording" to begin real-time conversion</li>
                    <li>Speak into your microphone</li>
                    <li>Converted audio will play automatically</li>
                    <li>Click "Stop Recording" when done</li>
                </ol>
            </div>
        </div>
    );
};

export default AudioRecorder;
