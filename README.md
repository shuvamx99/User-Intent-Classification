# Smart Fraud Detection System 🛡️

## What does it do?

This system helps you catch fraudulent users by automatically selecting the right detection tools based on your request. It gets smarter over time by learning from your feedback.

### Simple Example:

- **You ask:** "Find users with suspicious locations"
- **System:** Runs geographic and IP analysis tools
- **You get:** List of potentially fraudulent users
- **You rate:** How good were the results (1-5 stars)
- **System learns:** Uses your feedback to improve future recommendations

## How It Works

```mermaid
flowchart TD
    A[👤 You ask:<br/>"Find suspicious users"] --> B[🧠 AI understands<br/>your request]
    B --> C[📚 Checks memory:<br/>"What worked before?"]
    C --> D[🤖 Selects best tools:<br/>Geographic + IP + Speed]
    D --> E[🔍 Runs fraud detection<br/>on your data]
    E --> F[📊 Shows results:<br/>Suspicious users found]
    F --> G[⭐ You rate results<br/>1-5 stars]
    G --> H[💾 System learns<br/>and remembers]
    H -.-> C

    style A fill:#e3f2fd
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#e8f5e8
    style E fill:#fff9c4
    style F fill:#ffebee
    style G fill:#f1f8e9
    style H fill:#e3f2fd
```

## Available Detection Tools

| 🔍 Tool               | What it catches                         |
| --------------------- | --------------------------------------- |
| 🌍 Geographic Checker | Users appearing in impossible locations |
| 🌐 IP Analyzer        | Suspicious network patterns             |
| ⚡ Speed Detector     | Unnaturally fast user actions           |

## Quick Start

1. **Setup:** Add your OpenAI API key to `.env` file
2. **Start:** Run `docker-compose up -d`
3. **Use:** Run `docker-compose exec intent-classification-system python main.py`
4. **Ask:** Type requests like "check for location anomalies"
5. **Rate:** Give feedback to help the system learn

## Sample Questions You Can Ask

- "Find users with suspicious geographic activity"
- "Check for IP address anomalies"
- "Detect users completing actions too quickly"
- "Run all fraud detection tools"

That's it! The system handles the rest automatically.

---

## Technical Setup Details

### Prerequisites

- Docker and Docker Compose
- OpenAI API Key

### Environment Setup

Create a `.env` file:

```bash
API_KEY=your-openai-api-key-here
```

### Commands

```bash
# Start the system
docker-compose up -d

# Run the application
docker-compose exec intent-classification-system python main.py

# Check system status
docker-compose ps

# Stop the system
docker-compose down
```

### Data Storage

- All your data stays on your computer (Milvus database)
- System learns from your feedback locally

## Support

If something doesn't work:

1. Check `docker-compose ps` to ensure services are running
2. Look at logs: `docker-compose logs intent-classification-system`
3. Restart: `docker-compose restart`
