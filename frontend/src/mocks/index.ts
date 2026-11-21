/**
 * Punto de entrada de los fixtures del modo demo offline. El cliente lo importa de forma
 * diferida (`import("../mocks")`) para no cargarlo cuando la API está disponible.
 */
export { MOCK_CHUNKS, MOCK_DOCUMENTS, MOCK_PDF_SOURCES } from "./corpus";
export { MOCK_EVAL } from "./eval";
export { MOCK_SECURITY } from "./security";
export { buildMockResponse, confirmMockAction, matchScenario, mockUsers } from "./scenarios";
