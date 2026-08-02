import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ChatService } from './chat.service';

describe('ChatService', () => {
  let service: ChatService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(ChatService);
    http = TestBed.inject(HttpTestingController);
  });

  it('posts a typed chat request', () => {
    service.send({ message: 'What is OctaKidz?', session_id: 'session-1' }).subscribe();
    const request = http.expectOne('http://localhost:8000/api/chat');
    expect(request.request.method).toBe('POST');
    expect(request.request.body.message).toBe('What is OctaKidz?');
    request.flush({
      final_response: 'A grounded answer.',
      category: 'About OctaKidz',
      confidence: 'high',
      escalated: false,
      lead_capture_requested: false,
      internal_note: 'internal',
    });
  });
});

