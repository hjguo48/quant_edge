export type Status = 'PASS' | 'FAIL' | 'PENDING' | 'ACTIVE' | 'ALERT';

export interface User {
  name: string;
  role: string;
  avatar?: string;
}

export interface NavItem {
  name: string;
  path: string;
  icon: string;
}
