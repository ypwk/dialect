"use client";

import { usePathname } from "next/navigation";
import Link from "next/link";

export function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="bg-gray-800 text-white p-4">
      <ul className="flex justify-center space-x-4">
        <li>
          <Link href="/" passHref>
            <span
              className={`inline-block px-3 py-2 rounded-md text-sm font-medium cursor-pointer ${
                pathname === "/" ? "bg-gray-900" : "hover:bg-gray-700"
              }`}
            >
              Home
            </span>
          </Link>
        </li>
        <li>
          <Link href="/chat" passHref>
            <span
              className={`inline-block px-3 py-2 rounded-md text-sm font-medium cursor-pointer ${
                pathname === "/chat" ? "bg-gray-900" : "hover:bg-gray-700"
              }`}
            >
              Chat
            </span>
          </Link>
        </li>
        <li>
          <Link href="/about" passHref>
            <span
              className={`inline-block px-3 py-2 rounded-md text-sm font-medium cursor-pointer ${
                pathname === "/about" ? "bg-gray-900" : "hover:bg-gray-700"
              }`}
            >
              About
            </span>
          </Link>
        </li>
      </ul>
    </nav>
  );
}
