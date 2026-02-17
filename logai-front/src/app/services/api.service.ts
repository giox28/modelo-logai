import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private baseUrl = 'http://localhost:8001'; // Changed to 8001 to avoid port conflict

  constructor(private http: HttpClient) { }

  getAvailableModels(basin: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/available-models/${basin}`);
  }

  inspectFile(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post(`${this.baseUrl}/inspect-well`, formData);
  }

  processWell(formData: FormData): Observable<any> {
    return this.http.post(`${this.baseUrl}/process-well`, formData);
  }
}
