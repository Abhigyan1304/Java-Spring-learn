import { configureStore } from '@reduxjs/toolkit';
import chartReducer from './chartSlice';

export const store = configureStore({
  reducer: {
    chart: chartReducer, // Our chart slice reducer
  },
});

// Define RootState and AppDispatch types for TypeScript
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
