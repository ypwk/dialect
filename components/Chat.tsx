"use client";

import React, { useState, useEffect, useRef } from "react";
import io, { Socket } from "socket.io-client";

const SOCKET_URL = "http://127.0.0.1:5000/";

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<string[]>([]);
  const webSocketRef = useRef<Socket | null>(null);

  useEffect(() => {
    const socket = io(SOCKET_URL, {
      transports: ["websocket"],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });
    webSocketRef.current = socket;

    socket.on("connect_error", (err) => {
      console.log("Connection Error:", err.message);
    });

    socket.on("new_message", (message: string) => {
      setMessages((prevMessages) => [...prevMessages, message]);
    });

    socket.on("connect", () => {
      console.log("Connected to server");
      webSocketRef.current?.emit("join");
    });

    socket.on("new_philosopher", (phil: string) => {
      if (phil.token.trim() !== "") {
        setMessages((prevMessages) => [...prevMessages, phil.token.trim()]);
      }
      console.log(messages);
    });

    socket.on("new_token", (token: string) => {
      if (token.token.trim() !== "") {
        setMessages((prevMessages) => [
          ...prevMessages.slice(0, prevMessages.length - 1),
          prevMessages[prevMessages.length - 1] +
            " " +
            token.token.trim().replace(/<\/s>/g, ""),
        ]);
      }
      console.log(messages);
    });

    return () => {
      if (webSocketRef.current) {
        webSocketRef.current.emit("leave");
        webSocketRef.current.disconnect();
      }
    };
  });

  const sendMessage = (message: string) => {
    if (webSocketRef.current) {
      webSocketRef.current.emit("message", { message });
    }
  };

  return (
    <>
      <div
        style={{
          marginLeft: "5%",
          marginRight: "5%",
          marginTop: "2em",
          overflowY: "auto",
          maxHeight: "80vh",
          padding: "10px",
          border: "1px solid #ccc",
          borderRadius: "5px",
          display: "flex",
          flexDirection: "column", // Add this line
        }}
      >
        {messages.map((msg, index) => (
          <React.Fragment key={index}>
            {" "}
            {/* Change here */}
            <div style={{ width: "5%", padding: "10px", marginRight: "5px" }}>
              {msg.split(" ")[0]}
            </div>
            <div style={{ flex: 1, padding: "10px" }}>
              {msg.split(" ").slice(1).join(" ")}
            </div>
          </React.Fragment>
        ))}
      </div>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
        }}
      >
        {/* <ul style={{ listStyleType: "none", padding: 0, overflowY: "auto" }}>
          {messages.map((msg, index) => (
            <li key={index}>{msg}</li>
          ))}
        </ul> */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            const messageInput = e.currentTarget
              .elements[0] as HTMLInputElement;
            sendMessage(messageInput.value);
            messageInput.value = "";
          }}
          style={{ display: "flex", width: "90%" }}
        >
          <input
            type="text"
            placeholder="Type your message here..."
            style={{ flexGrow: 1, padding: "10px", color: "black" }}
          />
          <button
            style={{ backgroundColor: "grey", padding: "10px" }}
            type="submit"
          >
            Send
          </button>
        </form>
      </div>
    </>
  );
};

export default Chat;
