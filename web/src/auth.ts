// Google sign-in through Firebase Auth, only when the build is given a Firebase
// config. Without one the UI talks to a local server that asks for no sign-in.

import { initializeApp, type FirebaseApp } from "firebase/app";
import {
  GoogleAuthProvider, getAuth, onAuthStateChanged, signInWithPopup, signOut as fbSignOut,
  type Auth, type User,
} from "firebase/auth";

const env = import.meta.env;
const config = {
  apiKey: env.VITE_FIREBASE_API_KEY as string | undefined,
  authDomain: env.VITE_FIREBASE_AUTH_DOMAIN as string | undefined,
  projectId: env.VITE_FIREBASE_PROJECT_ID as string | undefined,
  appId: env.VITE_FIREBASE_APP_ID as string | undefined,
};

export const firebaseConfigured = Boolean(config.apiKey && config.projectId);

let app: FirebaseApp | null = null;
let auth: Auth | null = null;
if (firebaseConfigured) {
  app = initializeApp(config);
  auth = getAuth(app);
}

export function watchUser(cb: (u: User | null) => void): () => void {
  if (!auth) { cb(null); return () => {}; }
  return onAuthStateChanged(auth, cb);
}

export async function signIn(): Promise<void> {
  if (!auth) throw new Error("sign-in is not configured in this build");
  await signInWithPopup(auth, new GoogleAuthProvider());
}

export async function signOut(): Promise<void> {
  if (auth) await fbSignOut(auth);
}

/** The current user's ID token (refreshed by Firebase when near expiry), or null. */
export async function idToken(): Promise<string | null> {
  return auth?.currentUser ? auth.currentUser.getIdToken() : null;
}
