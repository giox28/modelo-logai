import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { map, Observable } from 'rxjs';

@Injectable({
    providedIn: 'root'
})
export class AuthService {
    private apiUrl = 'http://localhost:8003';

    constructor(private http: HttpClient) { }

    login(username: string, password: string): Observable<any> {
        const body = new URLSearchParams();
        body.set('username', username);
        body.set('password', password);

        const headers = new HttpHeaders({
            'Content-Type': 'application/x-www-form-urlencoded'
        });

        return this.http.post<any>(`${this.apiUrl}/token`, body.toString(), { headers }).pipe(
            map(response => {
                if (response && response.access_token) {
                    localStorage.setItem('access_token', response.access_token);
                    if (response.refresh_token) {
                        localStorage.setItem('refresh_token', response.refresh_token);
                    }
                    return true;
                }
                return false;
            })
        );
    }

    logout() {
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
    }

    public get isAuthenticated(): boolean {
        const token = localStorage.getItem('access_token');
        return !!token;
    }
}
