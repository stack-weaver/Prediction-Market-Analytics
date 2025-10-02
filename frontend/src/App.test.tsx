import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import App from './App';

describe('App', () => {
  it('renders stock prediction app', () => {
    render(<App />);
    const titleElement = screen.getByText(/Stock Price Prediction/i);
    expect(titleElement).toBeInTheDocument();
  });
});
