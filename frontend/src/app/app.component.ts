import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ChatMessage, ChatResponse } from './chat.models';
import { ChatService } from './chat.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent {
  readonly suggestedQuestions = [
    'What is OctaKidz?',
    'What age group is OctaKidz for?',
    'Are the activities screen-free?',
    'How long does one activity take?',
  ];

  messages: ChatMessage[] = [
    {
      role: 'assistant',
      text: 'Hello! I’m here to help with OctaKidz questions. Ask about the program, activities, ages, or support.',
    },
  ];
  draft = '';
  loading = false;
  errorMessage = '';
  showContactForm = false;
  contactName = '';
  contactEmail = '';

  private readonly sessionId = this.getOrCreateSessionId();

  constructor(private readonly chat: ChatService) {}

  send(): void {
    const message = this.draft.trim();
    if (!message || this.loading) {
      return;
    }

    this.messages = [...this.messages, { role: 'user', text: message }];
    this.draft = '';
    this.loading = true;
    this.errorMessage = '';
    this.showContactForm = false;

    this.chat.send({ message, session_id: this.sessionId }).subscribe({
      next: (response: ChatResponse) => {
        this.messages = [
          ...this.messages,
          {
            role: 'assistant',
            text: response.final_response,
            escalated: response.escalated,
            leadCaptureRequested: response.lead_capture_requested,
          },
        ];
        this.loading = false;
      },
      error: () => {
        this.errorMessage = 'We could not send your question right now. Please try again in a moment.';
        this.loading = false;
      },
    });
  }

  useSuggestion(question: string): void {
    this.draft = question;
  }

  openContactForm(): void {
    this.showContactForm = true;
  }

  private getOrCreateSessionId(): string {
    const key = 'octakidz-faq-session-id';
    const existing = localStorage.getItem(key);
    if (existing) {
      return existing;
    }
    const value = typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : 'session-' + Date.now() + '-' + Math.random().toString(16).slice(2);
    localStorage.setItem(key, value);
    return value;
  }
}

