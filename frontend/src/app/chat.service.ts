import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../environments/environment';
import { ChatRequest, ChatResponse } from './chat.models';

@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly http = inject(HttpClient);

  send(request: ChatRequest): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(environment.apiBaseUrl + '/api/chat', request);
  }
}

