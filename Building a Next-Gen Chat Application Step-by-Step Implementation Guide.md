You are tasked with creating a the code for a web application that emulates the core functionality of ChatGPT.com. Your goal is to provide clear, step-by-step instruction and code for me to follow in order to build this application. Here are the project requirements and technology stack:

<project_requirements>

1. User Experience:
   - Design a modern, minimalist chat interface inspired by popular messaging apps
   - Implement smooth transitions and animations for a polished feel
   - Create a dark mode option for improved readability in low-light conditions
   - Develop a customizable interface with user-defined themes and layouts

2. Conversation Management:
   - Enable users to organize conversations into folders or categories
   - Implement a search function to quickly find specific messages or conversations
   - Allow users to share conversation snippets with unique, shareable links
   - Provide options to export entire conversations in various formats (PDF, TXT, JSON)

3. AI Interaction:
   - Integrate with Claude-3.5-Sonnet API for advanced language understanding and generation
   - Implement context-aware responses that maintain coherence across long conversations
   - Allow users to adjust AI personality traits (e.g., formal, casual, creative, analytical)
   - Provide real-time suggestions and auto-completions as users type

4. User Accounts:
   - Implement a secure, multi-factor authentication system
   - Create a user profile system with customizable avatars and bios
   - Develop a referral program to incentivize user growth
   - Implement a tiered subscription model with different feature sets

5. Data Handling:
   - Implement end-to-end encryption for all user conversations
   - Provide users with options to delete their data or request data exports
   - Use data anonymization techniques for improving AI models while preserving privacy
   - Implement automatic data backup and recovery systems

6. Performance Optimization:
   - Utilize lazy loading and code splitting to minimize initial load times
   - Implement a content delivery network (CDN) for faster global access
   - Optimize database queries with proper indexing and caching strategies
   - Use WebAssembly for computationally intensive tasks to improve performance

7. Security Measures:
   - Implement CAPTCHA or similar systems to prevent bot abuse
   - Conduct regular security audits and penetration testing
   - Use secure WebSocket connections for real-time communication
   - Implement a bug bounty program to encourage responsible disclosure of vulnerabilities

8. Scalability and Reliability:
   - Design a microservices architecture for easier scaling of individual components
   - Implement auto-scaling based on server load and user demand
   - Use database sharding for improved performance with large datasets
   - Implement a robust error logging and monitoring system

9. Business Intelligence:
   - Develop a comprehensive analytics dashboard for tracking key performance indicators
   - Implement A/B testing capabilities for new features and UI changes
   - Create a system for collecting and analyzing user feedback and feature requests
   - Develop predictive models for user churn and engagement

10. Accessibility Features:
    - Ensure full keyboard navigation support throughout the application
    - Implement high-contrast modes and adjustable font sizes for improved readability
    - Provide audio descriptions and transcripts for any multimedia content
    - Conduct regular accessibility audits to maintain WCAG 2.1 Level AA compliance

11. Localization:
    - Implement a robust internationalization system supporting right-to-left languages
    - Provide region-specific content and features based on user location
    - Allow community-driven translations to expand language support efficiently
    - Implement currency conversion for any paid features or subscriptions

12. Third-party Integrations:
    - Develop plugins for popular productivity tools (e.g., Slack, Microsoft Teams)
    - Create an API for third-party developers to build on top of the platform
    - Implement social media sharing capabilities for conversation snippets
    - Integrate with calendar apps for scheduling and reminders based on chat content



<tech_stack>

Frontend:
- Next.js for server-side rendering and improved SEO
- MobX for state management
- Emotion for CSS-in-JS styling
- GraphQL with Apollo Client for efficient data fetching
- Pusher for real-time features

Backend:
- Nest.js for scalable and maintainable server-side architecture
- GraphQL with Apollo Server for flexible API development
- Passport.js with OAuth 2.0 for authentication
- JSON Web Tokens (JWT) for stateless authentication

Database:
- PostgreSQL for primary relational data storage
- Cassandra for handling large-scale, distributed data
- Redis for caching and session management

AI Integration:
- Claude-3.5-Sonnet API for natural language processing
- TensorFlow.js for client-side machine learning capabilities

Testing:
- Jest and React Testing Library for unit and integration testing
- Cypress for end-to-end testing
- Lighthouse for performance and accessibility testing

DevOps:
- Docker for containerization
- Kubernetes for container orchestration
- GitLab CI/CD for continuous integration and deployment

Monitoring and Logging:
- Datadog for comprehensive monitoring and logging
- Sentry for real-time error tracking and debugging

Security:
- HTTPS with AWS Certificate Manager for SSL/TLS
- Argon2 for secure password hashing
- helmet.js for setting secure HTTP headers
- OAuth 2.0 for third-party authentication

Deployment:
- Amazon Web Services (AWS) for cloud hosting
- Amazon EKS (Elastic Kubernetes Service) for container orchestration
- Amazon S3 for object storage and CDN integration

Other Tools:
- Webpack 5 for module bundling
- TypeScript for static typing
- ESLint and Prettier for code linting and formatting
- Storybook for component development and documentation
- Postman for API testing and documentation

Performance Optimization:
- Cloudflare Workers for edge computing capabilities
- WebAssembly (WASM) for high-performance computing tasks

Internationalization:
- react-intl for managing translations and localized content

Accessibility:
- axe-core for automated accessibility testing

Version Control:
- Git with GitHub for source code management and collaboration

Please create a comprehensive project plan and implementation guide based on the given requirements and technology stack. Your response should include the following sections:

1. Project Overview

Provide a brief summary of the project, its main objectives, and key features.

2. Architecture Design

Describe the high-level architecture of the application, including the frontend, backend, database, and any external services or APIs.

3. Development Phases

Break down the project into logical development phases, each with specific goals and deliverables.

4. Detailed Implementation Guide

For each phase, provide step-by-step instructions on how to implement the required features. Include code snippets, best practices, and potential challenges to consider.

5. Testing Strategy

Outline a comprehensive testing strategy, including unit tests, integration tests, and end-to-end tests.

6. Security Considerations

Provide guidelines for implementing security best practices, including user authentication, data encryption, and API key management.

7. Performance Optimization

Describe techniques for optimizing the application's performance, including caching strategies and database query optimization.

8. Deployment and Maintenance

Provide instructions for deploying the application to a production environment and maintaining it over time.

9. Documentation

Outline the required documentation for the project, including API documentation, user guides, and developer documentation.

10. Timeline and Milestones

Propose a realistic timeline for the project, including major milestones and deliverables.

Please format your response using appropriate headings and subheadings for easy readability. Include code snippets where necessary, enclosed in <code> tags. If you need to make any assumptions about the project or technology stack, please state them clearly.

Begin your response with:

<project_plan>

And end your response with:

</project_plan>

Remember to provide detailed, actionable instructions that a development team can follow to successfully implement this web application.

