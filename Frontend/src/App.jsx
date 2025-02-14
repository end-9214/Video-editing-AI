import React, { useState } from 'react';
import './App.css';

function App() {
  const [video, setVideo] = useState(null);
  const [chatMessages, setChatMessages] = useState([]);
  const [autoEditOptionsVisible, setAutoEditOptionsVisible] = useState(false);

  const handleVideoUpload = (event) => {
    const file = event.target.files[0];
    setVideo(URL.createObjectURL(file));
  };

  const handleChatMessageSubmit = (message) => {
    setChatMessages([...chatMessages, { text: message, sender: 'user' }]);
    // Here, you would typically send the message to an AI backend
    // and process the response. For now, let's simulate a response.
    setTimeout(() => {
      setChatMessages(prevMessages => [...prevMessages, { text: `AI: Processing "${message}"...`, sender: 'ai' }]);
    }, 1000);
  };

  const handleAutoEditClick = () => {
    setAutoEditOptionsVisible(!autoEditOptionsVisible);
  };

  const handleAutoEditOptionClick = (option) => {
    setChatMessages([...chatMessages, { text: `User: Auto ${option}`, sender: 'user' }]);
    setTimeout(() => {
      setChatMessages(prevMessages => [...prevMessages, { text: `AI: Applying Auto ${option}...`, sender: 'ai' }]);
    }, 1000);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>AI Video Editor</h1>
      </header>

      <main className="app-main">
        <section className="upload-section">
          <input type="file" accept="video/*" onChange={handleVideoUpload} className="upload-input" />
          {video && (
            <video width="400" controls className="video-preview">
              <source src={video} type="video/mp4" />
              Your browser does not support HTML5 video.
            </video>
          )}
        </section>

        <section className="editing-area">
          <div className="chat-interface">
            <div className="chat-messages">
              {chatMessages.map((message, index) => (
                <div key={index} className={`message ${message.sender}`}>
                  {message.text}
                </div>
              ))}
            </div>
            <ChatInput onMessageSubmit={handleChatMessageSubmit} />
          </div>

          <div className="auto-edit-panel">
            <button onClick={handleAutoEditClick} className="auto-edit-button">
              Auto Edit
            </button>
            {autoEditOptionsVisible && (
              <div className="auto-edit-options">
                <button onClick={() => handleAutoEditOptionClick('Crop')}>Auto Crop</button>
                <button onClick={() => handleAutoEditOptionClick('Trim')}>Auto Trim</button>
                <button onClick={() => handleAutoEditOptionClick('Style')}>Auto Style</button>
                <button onClick={() => handleAutoEditOptionClick('Stabilize')}>Auto Stabilize</button>
                <button onClick={() => handleAutoEditOptionClick('Enhance')}>Auto Enhance</button>
                <button onClick={() => handleAutoEditOptionClick('Add Captions')}>Auto Add Captions</button>
              </div>
            )}
          </div>
        </section>
      </main>

      <footer className="export-section">
        <button className="export-button">Export Video</button>
      </footer>
    </div>
  );
}

function ChatInput({ onMessageSubmit }) {
  const [message, setMessage] = useState('');

  const handleSubmit = (event) => {
    event.preventDefault();
    onMessageSubmit(message);
    setMessage('');
  };

  return (
    <form onSubmit={handleSubmit} className="chat-input">
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Type your message..."
        className="chat-input-field"
      />
      <button type="submit" className="chat-input-button">Send</button>
    </form>
  );
}

export default App;
