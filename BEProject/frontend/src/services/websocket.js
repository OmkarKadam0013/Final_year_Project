// frontend/src/services/websocket.js
class AudioStreamService {
    constructor() {
        this.ws = null;
        this.clientId = this.generateClientId();
        this.audioContext = null;
        this.mediaRecorder = null;
        this.isRecording = false;
        this.messageHandlers = {};
        
        this.config = {
            sampleRate: 16000,
            chunkSize: 4096,
            channels: 1
        };
    }
    
    generateClientId() {
        return 'client_' + Math.random().toString(36).substr(2, 9);
    }
    
    async connect(serverUrl) {
        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(`${serverUrl}/ws/${this.clientId}`);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                resolve();
            };
            
            this.ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                this.handleMessage(message);
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                reject(error);
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket closed');
            };
        });
    }
    
    handleMessage(message) {
        const handler = this.messageHandlers[message.type];
        if (handler) {
            handler(message);
        }
    }
    
    on(messageType, handler) {
        this.messageHandlers[messageType] = handler;
    }
    
    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: this.config.sampleRate,
                    channelCount: this.config.channels,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });
            
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.config.sampleRate
            });
            
            const source = this.audioContext.createMediaStreamSource(stream);
            const processor = this.audioContext.createScriptProcessor(this.config.chunkSize, 1, 1);
            
            processor.onaudioprocess = (e) => {
                if (this.isRecording) {
                    const inputData = e.inputBuffer.getChannelData(0);
                    this.sendAudioChunk(inputData);
                }
            };
            
            source.connect(processor);
            processor.connect(this.audioContext.destination);
            
            this.isRecording = true;
            console.log('Recording started');
            
        } catch (error) {
            console.error('Error starting recording:', error);
            throw error;
        }
    }
    
    stopRecording() {
        this.isRecording = false;
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        console.log('Recording stopped');
    }
    
    sendAudioChunk(audioData) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            // Convert Float32Array to Int16Array
            const int16Array = new Int16Array(audioData.length);
            for (let i = 0; i < audioData.length; i++) {
                int16Array[i] = Math.max(-32768, Math.min(32767, audioData[i] * 32768));
            }
            
            // Convert to base64
            const base64Audio = this.arrayBufferToBase64(int16Array.buffer);
            
            // Send
            this.ws.send(JSON.stringify({
                type: 'audio',
                data: base64Audio
            }));
        }
    }
    
    arrayBufferToBase64(buffer) {
        let binary = '';
        const bytes = new Uint8Array(buffer);
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return window.btoa(binary);
    }
    
    base64ToArrayBuffer(base64) {
        const binaryString = window.atob(base64);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        return bytes.buffer;
    }
    
    playAudio(base64Audio) {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: this.config.sampleRate
        });
        
        const arrayBuffer = this.base64ToArrayBuffer(base64Audio);
        const int16Array = new Int16Array(arrayBuffer);
        const float32Array = new Float32Array(int16Array.length);
        
        // Convert Int16 to Float32
        for (let i = 0; i < int16Array.length; i++) {
            float32Array[i] = int16Array[i] / 32768.0;
        }
        
        const audioBuffer = audioContext.createBuffer(1, float32Array.length, this.config.sampleRate);
        audioBuffer.getChannelData(0).set(float32Array);
        
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        source.start();
    }
    
    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.stopRecording();
    }
}

export default AudioStreamService;
