# main.py
from api_manager import APIManager
from embedding_saver import EmbeddingSaver

def main():
    api_key = "Upstage api key 입력"  # 실제 API 키로 교체
    api_manager = APIManager(api_key=api_key)
    
    embedding_saver = EmbeddingSaver(api_key=api_key)
    
    # 경로 및 파일 이름 설정
    embedding_saver.set_dir_path("저장경로", "저장폴더이름", "저장파일이름")
    
    # 데이터 로드
    try:
        docs = embedding_saver.load_data()
        print(f"로드된 문서의 개수: {len(docs)}")
        
        # 임베딩 생성
        embedding_saver.create_embeddings()
        
        # 임베딩 저장
        embedding_saver.save_embeddings(docs)
    except Exception as e:
        print(f"문서 로드 중 오류 발생: {e}")

if __name__ == "__main__":
    main()