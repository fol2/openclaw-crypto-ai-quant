import App from './App.svelte';
import { mount } from 'svelte';

// Register Lit web components (when available).
// import './wc/index';

const app = mount(App, { target: document.getElementById('app')! });

export default app;
