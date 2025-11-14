// frontend/src/App.js
import React from 'react';
import './App.css';
import AudioRecorder from './components/AudioRecorder';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Dysarthric Speech Conversion</h1>
        <p>Real-time conversion of dysarthric speech to clear speech</p>
      </header>
      
      <main className="App-main">
        <AudioRecorder />
      </main>
      
      <footer className="App-footer">
        <p>© 2025 Dysarthric Speech Conversion System | Powered by AI</p>
        <p>
          <a href="https://github.com/pravi7072/dysarthric-speech-conversion" 
             target="_blank" 
             rel="noopener noreferrer">
            GitHub
          </a>
          {' | '}
          <a href="/docs" target="_blank" rel="noopener noreferrer">
            Documentation
          </a>
        </p>
      </footer>
    </div>
  );
}

export default App;
